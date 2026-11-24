// XGameUi : demandes d'interface confiées aux rappels du jeu (voir gameui_requests.h).
//
// Chaque demande est une opération XAsync : Begin la planifie sur la file du jeu, DoWork
// appelle le rappel du jeu et laisse l'opération en cours (E_PENDING), puis la réponse du jeu
// (XGameUiSet*UiResponse) ou une annulation la termine. La poignée remise au jeu est l'adresse
// de la demande ; elle n'est acceptée qu'une fois, et seulement tant que la demande attend.
#include "gameui_requests.h"

#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_set>
#include <utility>

#include "XAsyncProvider.h"
#include "trace.h"

namespace anyxboxx {
namespace {

struct UiRequest {
  UiKind kind;
  // Partagé avec DoWork : les textes passés au jeu restent valables pendant son rappel,
  // même si la réponse arrive d'un autre fil avant que le rappel ne rende la main.
  std::shared_ptr<UiInvoke> invoke;
  XAsyncBlock* async = nullptr;  // bloc du fournisseur (XAsyncProviderData::async)
  void* game_queue = nullptr;    // file donnée par le jeu dans son XAsyncBlock
  std::vector<uint8_t> result;
};

struct UiState {
  std::mutex mutex;
  UiCallbacks callbacks{};
  std::unordered_set<UiRequest*> waiting;  // demandes lancées, ni répondues ni annulées
};

UiState& State() {
  static UiState state;
  return state;
}

void Wait(UiRequest* request) {
  std::lock_guard<std::mutex> lock(State().mutex);
  State().waiting.insert(request);
}

// Retire la demande de l'attente ; vrai pour un seul appelant (réponse, annulation ou fin).
bool Claim(UiRequest* request) {
  std::lock_guard<std::mutex> lock(State().mutex);
  return State().waiting.erase(request) == 1;
}

bool Waiting(UiRequest* request) {
  std::lock_guard<std::mutex> lock(State().mutex);
  return State().waiting.count(request) == 1;
}

HRESULT CALLBACK UiRequestProvider(XAsyncOp op, const XAsyncProviderData* data) noexcept {
  auto* request = static_cast<UiRequest*>(data->context);
  switch (op) {
    case XAsyncOp::Begin:
      request->async = data->async;
      Wait(request);
      return XAsyncSchedule(data->async, 0);
    case XAsyncOp::DoWork: {
      if (!Waiting(request)) return E_ABORT;  // annulée avant d'atteindre le jeu
      const std::shared_ptr<UiInvoke> invoke = request->invoke;
      // La file du jeu telle qu'il l'a donnée (la copie interne en tient un double).
      (*invoke)(request, request->game_queue != nullptr ? request->game_queue : data->async->queue);
      return E_PENDING;  // terminée par la réponse du jeu
    }
    case XAsyncOp::GetResult:
      if (!request->result.empty()) std::memcpy(data->buffer, request->result.data(), request->result.size());
      return S_OK;
    case XAsyncOp::Cancel:
      if (Claim(request)) XAsyncComplete(data->async, E_ABORT, 0);
      return S_OK;
    case XAsyncOp::Cleanup:
      Claim(request);
      delete request;
      return S_OK;
    default:
      return S_OK;
  }
}

}  // namespace

void SetUiCallbacks(const UiCallbacks* callbacks) {
  std::lock_guard<std::mutex> lock(State().mutex);
  State().callbacks = callbacks != nullptr ? *callbacks : UiCallbacks{};
}

bool FindUiCallback(UiKind kind, void** callback, void** context) {
  std::lock_guard<std::mutex> lock(State().mutex);
  *callback = State().callbacks.callbacks[static_cast<uint32_t>(kind)];
  *context = State().callbacks.context;
  return *callback != nullptr;
}

HRESULT BeginUiRequest(XAsyncBlock* async, const void* identity, const char* name, UiKind kind, UiInvoke invoke) {
  if (async == nullptr) return E_INVALIDARG;
  auto* request = new (std::nothrow) UiRequest{kind, nullptr};
  if (request == nullptr) return E_OUTOFMEMORY;
  request->invoke = std::make_shared<UiInvoke>(std::move(invoke));
  request->game_queue = async->queue;
  const HRESULT begun = XAsyncBegin(async, request, identity, name, UiRequestProvider);
  if (FAILED(begun)) delete request;  // le fournisseur n'a jamais été appelé
  ANYXBOXX_TRACE("[runtime] %s : demande confiée au rappel du jeu", name);
  return begun;
}

HRESULT CompleteUiRequest(void* handle, UiKind kind, std::vector<uint8_t> result) {
  auto* request = static_cast<UiRequest*>(handle);
  XAsyncBlock* async = nullptr;
  {
    std::lock_guard<std::mutex> lock(State().mutex);
    auto found = State().waiting.find(request);
    if (found == State().waiting.end() || request->kind != kind) return E_INVALIDARG;
    State().waiting.erase(found);
    request->result = std::move(result);
    async = request->async;
  }
  const size_t size = request->result.size();
  XAsyncComplete(async, S_OK, size);  // sans résultat, la demande peut être libérée ici
  return S_OK;
}

}  // namespace anyxboxx
