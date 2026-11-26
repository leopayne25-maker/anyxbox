// XGameUi sans interface système : les écrans sont confiés aux rappels du jeu.
//
// Méthodes : gdk_api_table.inc. Signatures : pages XGameUi de Microsoft Learn,
// https://learn.microsoft.com/gaming/gdk/docs/reference/system/xgameui/xgameui_members
// (une page par fonction sous .../xgameui/functions/<nom en minuscules>).
// Sans Gaming Services, AnyXboxX n'a pas d'interface système à montrer. Si le jeu a fourni
// ses propres écrans (XGameUiSetUiCallbacks), chaque XGameUiShow*Async appelle le rappel du
// jeu et rend la réponse que ce rappel donne par XGameUiSet*UiResponse. Sinon, l'appel
// échoue en E_NOTIMPL : aucune réponse de l'utilisateur n'est inventée.
#include <windows.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "XAsync.h"
#include "XAsyncProvider.h"
#include "gameui_requests.h"
#include "interface_helpers.h"
#include "interfaces.h"
#include "trace.h"

namespace {

using anyxboxx::UiKind;

// Texte facultatif copié : le jeu peut libérer le sien dès le retour de *Async.
struct Text {
  bool present = false;
  std::string value;
  const char* get() const { return present ? value.c_str() : nullptr; }
};
Text Copy(const char* text) { return text != nullptr ? Text{true, text} : Text{}; }

// XGameUiMessageDialogButton : First, Second, Third.
constexpr uint32_t kLastButton = 2;

// Identités des résultats : XAsyncGetResult vérifie qu'on lit ce que l'appel a produit.
const char kMessageIdentity = 0, kInviteIdentity = 0, kProfileIdentity = 0, kAchievementsIdentity = 0,
           kPickerIdentity = 0, kErrorIdentity = 0, kTextIdentity = 0, kActivityIdentity = 0;

// Rappel du jeu pour `kind`, ou trace et faux s'il n'y en a pas (pas d'interface système).
template <typename Callback>
bool GameCallback(UiKind kind, const char* name, Callback* callback, void** context) {
  void* raw = nullptr;
  if (!anyxboxx::FindUiCallback(kind, &raw, context)) {
    ANYXBOXX_TRACE("[runtime] %s : pas d'interface système sans Gaming Services -> E_NOTIMPL", name);
    return false;
  }
  *callback = reinterpret_cast<Callback>(raw);
  return true;
}

// --- Écrans sans résultat : profil, invitations, succès, erreur ---------------------------
using ProfileCardCallback = void(CALLBACK*)(void*, void*, void*, uint64_t, void*);
HRESULT ShowPlayerProfileCardAsync(XAsyncBlock* async, void* user, uint64_t target) noexcept {
  ProfileCardCallback callback = nullptr;
  void* context = nullptr;
  if (!GameCallback(UiKind::PlayerProfileCard, "XGameUiShowPlayerProfileCardAsync", &callback, &context)) return E_NOTIMPL;
  return anyxboxx::BeginUiRequest(async, &kProfileIdentity, "XGameUiShowPlayerProfileCardAsync", UiKind::PlayerProfileCard,
                                  [=](void* handle, void* queue) { callback(handle, queue, user, target, context); });
}

using SendInviteCallback = void(CALLBACK*)(void*, void*, void*, const char*, const char*, const char*, const char*,
                                           const char*, void*);
HRESULT ShowSendGameInviteAsync(XAsyncBlock* async, void* user, const char* scid, const char* template_name,
                                const char* session, const char* text, const char* activation) noexcept {
  SendInviteCallback callback = nullptr;
  void* context = nullptr;
  if (!GameCallback(UiKind::SendGameInvite, "XGameUiShowSendGameInviteAsync", &callback, &context)) return E_NOTIMPL;
  return anyxboxx::BeginUiRequest(
      async, &kInviteIdentity, "XGameUiShowSendGameInviteAsync", UiKind::SendGameInvite,
      [=, a = Copy(scid), b = Copy(template_name), c = Copy(session), d = Copy(text), e = Copy(activation)](
          void* handle, void* queue) { callback(handle, queue, user, a.get(), b.get(), c.get(), d.get(), e.get(), context); });
}

using AchievementsCallback = void(CALLBACK*)(void*, void*, void*, uint32_t, void*);
HRESULT ShowAchievementsAsync(XAsyncBlock* async, void* user, uint32_t title_id) noexcept {
  AchievementsCallback callback = nullptr;
  void* context = nullptr;
  if (!GameCallback(UiKind::Achievements, "XGameUiShowAchievementsAsync", &callback, &context)) return E_NOTIMPL;
  return anyxboxx::BeginUiRequest(async, &kAchievementsIdentity, "XGameUiShowAchievementsAsync", UiKind::Achievements,
                                  [=](void* handle, void* queue) { callback(handle, queue, user, title_id, context); });
}

using ActivityInviteCallback = void(CALLBACK*)(void*, void*, void*, void*);
HRESULT ShowMultiplayerActivityGameInviteAsync(XAsyncBlock* async, void* user) noexcept {
  ActivityInviteCallback callback = nullptr;
  void* context = nullptr;
  const char* name = "XGameUiShowMultiplayerActivityGameInviteAsync";
  if (!GameCallback(UiKind::MultiplayerActivityGameInvite, name, &callback, &context)) return E_NOTIMPL;
  return anyxboxx::BeginUiRequest(async, &kActivityIdentity, name, UiKind::MultiplayerActivityGameInvite,
                                  [=](void* handle, void* queue) { callback(handle, queue, user, context); });
}

using ErrorDialogCallback = void(CALLBACK*)(void*, void*, HRESULT, const char*, void*);
HRESULT ShowErrorDialogAsync(XAsyncBlock* async, HRESULT error, const char* service_context) noexcept {
  ErrorDialogCallback callback = nullptr;
  void* context = nullptr;
  if (!GameCallback(UiKind::ErrorDialog, "XGameUiShowErrorDialogAsync", &callback, &context)) return E_NOTIMPL;
  return anyxboxx::BeginUiRequest(
      async, &kErrorIdentity, "XGameUiShowErrorDialogAsync", UiKind::ErrorDialog,
      [=, text = Copy(service_context)](void* handle, void* queue) { callback(handle, queue, error, text.get(), context); });
}

// Fin d'un écran sans résultat : statut de l'opération (rien à copier).
HRESULT NoPayloadResult(XAsyncBlock* async, const char* identity) {
  return XAsyncGetResult(async, identity, 0, nullptr, nullptr);
}
HRESULT ShowPlayerProfileCardResult(XAsyncBlock* async) noexcept { return NoPayloadResult(async, &kProfileIdentity); }
HRESULT ShowSendGameInviteResult(XAsyncBlock* async) noexcept { return NoPayloadResult(async, &kInviteIdentity); }
HRESULT ShowAchievementsResult(XAsyncBlock* async) noexcept { return NoPayloadResult(async, &kAchievementsIdentity); }
HRESULT ShowActivityInviteResult(XAsyncBlock* async) noexcept { return NoPayloadResult(async, &kActivityIdentity); }
HRESULT ShowErrorDialogResult(XAsyncBlock* async) noexcept { return NoPayloadResult(async, &kErrorIdentity); }

// Réponses sans valeur : l'écran du jeu a été fermé.
template <UiKind Kind>
HRESULT DismissedResponse(void* handle) noexcept {
  return anyxboxx::CompleteUiRequest(handle, Kind, {});
}

// --- Boîte de message --------------------------------------------------------------------
using MessageDialogCallback = void(CALLBACK*)(void*, void*, const char*, const char*, const char*, const char*,
                                              const char*, uint32_t, uint32_t, void*);
HRESULT ShowMessageDialogAsync(XAsyncBlock* async, const char* title, const char* content, const char* first,
                               const char* second, const char* third, uint32_t default_button,
                               uint32_t cancel_button) noexcept {
  MessageDialogCallback callback = nullptr;
  void* context = nullptr;
  if (!GameCallback(UiKind::MessageDialog, "XGameUiShowMessageDialogAsync", &callback, &context)) return E_NOTIMPL;
  return anyxboxx::BeginUiRequest(
      async, &kMessageIdentity, "XGameUiShowMessageDialogAsync", UiKind::MessageDialog,
      [=, t = Copy(title), c = Copy(content), b1 = Copy(first), b2 = Copy(second), b3 = Copy(third)](
          void* handle, void* queue) {
        callback(handle, queue, t.get(), c.get(), b1.get(), b2.get(), b3.get(), default_button, cancel_button, context);
      });
}
HRESULT ShowMessageDialogResult(XAsyncBlock* async, uint32_t* button) noexcept {
  return XAsyncGetResult(async, &kMessageIdentity, sizeof *button, button, nullptr);
}
HRESULT SetMessageDialogUiResponse(void* handle, uint32_t button) noexcept {
  if (button > kLastButton) return E_INVALIDARG;
  std::vector<uint8_t> result(sizeof button);
  std::memcpy(result.data(), &button, sizeof button);
  return anyxboxx::CompleteUiRequest(handle, UiKind::MessageDialog, std::move(result));
}

// --- Saisie de texte : le résultat est la chaîne UTF-8 du jeu, zéro final compris ----------
using TextEntryCallback = void(CALLBACK*)(void*, void*, const char*, const char*, const char*, uint32_t, uint32_t,
                                          void*);
HRESULT ShowTextEntryAsync(XAsyncBlock* async, const char* title, const char* description, const char* initial,
                           uint32_t scope, uint32_t max_length) noexcept {
  TextEntryCallback callback = nullptr;
  void* context = nullptr;
  if (!GameCallback(UiKind::TextEntry, "XGameUiShowTextEntryAsync", &callback, &context)) return E_NOTIMPL;
  return anyxboxx::BeginUiRequest(
      async, &kTextIdentity, "XGameUiShowTextEntryAsync", UiKind::TextEntry,
      [=, t = Copy(title), d = Copy(description), i = Copy(initial)](void* handle, void* queue) {
        callback(handle, queue, t.get(), d.get(), i.get(), scope, max_length, context);
      });
}
HRESULT ShowTextEntryResultSize(XAsyncBlock* async, uint32_t* size) noexcept {
  if (size == nullptr) return E_INVALIDARG;
  size_t bytes = 0;
  const HRESULT status = XAsyncGetResultSize(async, &bytes);
  *size = static_cast<uint32_t>(bytes);
  return status;
}
HRESULT ShowTextEntryResult(XAsyncBlock* async, uint32_t size, char* buffer, uint32_t* used) noexcept {
  size_t bytes = 0;
  const HRESULT result = XAsyncGetResult(async, &kTextIdentity, size, buffer, &bytes);
  if (used != nullptr) *used = static_cast<uint32_t>(bytes);
  return result;
}
HRESULT SetTextEntryUiResponse(void* handle, const char* response) noexcept {
  if (response == nullptr) return E_INVALIDARG;
  const char* end = response + std::strlen(response) + 1;
  return anyxboxx::CompleteUiRequest(handle, UiKind::TextEntry, std::vector<uint8_t>(response, end));
}

// --- Choix de joueurs : XGameUiPlayerPickerInfo (structs/xgameuiplayerpickerinfo) ----------
struct PlayerPickerInfo {
  void* requesting_user;
  const char* prompt_text;
  uint32_t select_from_count;
  const uint64_t* select_from;
  uint32_t preselected_count;
  const uint64_t* preselected;
  uint32_t min_selection;
  uint32_t max_selection;
};
using PlayerPickerCallback = void(CALLBACK*)(void*, void*, PlayerPickerInfo*, void*);

HRESULT ShowPlayerPickerAsync(XAsyncBlock* async, void* user, const char* prompt, uint32_t from_count,
                              const uint64_t* from, uint32_t preselected_count, const uint64_t* preselected,
                              uint32_t min_selection, uint32_t max_selection) noexcept {
  if ((from_count > 0 && from == nullptr) || (preselected_count > 0 && preselected == nullptr)) return E_INVALIDARG;
  PlayerPickerCallback callback = nullptr;
  void* context = nullptr;
  if (!GameCallback(UiKind::PlayerPicker, "XGameUiShowPlayerPickerAsync", &callback, &context)) return E_NOTIMPL;
  std::vector<uint64_t> players(from, from + from_count), chosen(preselected, preselected + preselected_count);
  return anyxboxx::BeginUiRequest(
      async, &kPickerIdentity, "XGameUiShowPlayerPickerAsync", UiKind::PlayerPicker,
      [=, text = Copy(prompt), info = PlayerPickerInfo{}](void* handle, void* queue) mutable {
        info = PlayerPickerInfo{user, text.get(), from_count, players.data(), preselected_count, chosen.data(),
                                min_selection, max_selection};
        callback(handle, queue, &info, context);
      });
}
HRESULT ShowPlayerPickerResultCount(XAsyncBlock* async, uint32_t* count) noexcept {
  if (count == nullptr) return E_INVALIDARG;
  size_t bytes = 0;
  const HRESULT status = XAsyncGetResultSize(async, &bytes);
  *count = static_cast<uint32_t>(bytes / sizeof(uint64_t));
  return status;
}
HRESULT ShowPlayerPickerResult(XAsyncBlock* async, uint32_t count, uint64_t* players, uint32_t* used) noexcept {
  size_t bytes = 0;
  const HRESULT result = XAsyncGetResult(async, &kPickerIdentity, count * sizeof(uint64_t), players, &bytes);
  if (used != nullptr) *used = static_cast<uint32_t>(bytes / sizeof(uint64_t));
  return result;
}
HRESULT SetPlayerPickerUiResponse(void* handle, uint32_t count, const uint64_t* players) noexcept {
  if (count > 0 && players == nullptr) return E_INVALIDARG;
  const auto* bytes = reinterpret_cast<const uint8_t*>(players);
  return anyxboxx::CompleteUiRequest(handle, UiKind::PlayerPicker,
                                     std::vector<uint8_t>(bytes, bytes + count * sizeof(uint64_t)));
}

// --- Réglages ----------------------------------------------------------------------------
// « useSystemUiIfAvailable » : aucune interface système ici, les rappels du jeu servent toujours.
HRESULT SetUiCallbacks(const anyxboxx::UiCallbacks* callbacks, bool) noexcept {
  anyxboxx::SetUiCallbacks(callbacks);
  return S_OK;
}
// « This API exists on Windows, however it currently has no effect on achievement and other
// toast notifications. » (functions/xgameuisetnotificationpositionhint) : même chose ici.
HRESULT SetNotificationPositionHint(uint32_t position) noexcept {
  ANYXBOXX_TRACE("[runtime] XGameUiSetNotificationPositionHint(%u) : sans effet sur PC", position);
  return S_OK;
}

// Méthodes 3 à 50 de l'interface {EAF669DF…} (gdk_api_table.inc). Sans interface système ni
// rappel prévu par le GDK, l'authentification web, l'espace disque, le partage d'état et le
// clavier virtuel (XGameUiTextEntry*) restent non écrits.
void* const kGameUiMethods[] = {
    ANYXBOXX_IUNKNOWN_METHODS,                                     // 0-2
    ANYXBOXX_METHOD(ShowMessageDialogAsync),                       // 3
    ANYXBOXX_METHOD(ShowMessageDialogResult),                      // 4
    ANYXBOXX_METHOD(ShowSendGameInviteAsync),                      // 5
    ANYXBOXX_METHOD(ShowSendGameInviteResult),                     // 6
    ANYXBOXX_METHOD(ShowPlayerProfileCardAsync),                   // 7
    ANYXBOXX_METHOD(ShowPlayerProfileCardResult),                  // 8
    ANYXBOXX_METHOD(ShowAchievementsAsync),                        // 9
    ANYXBOXX_METHOD(ShowAchievementsResult),                       // 10
    ANYXBOXX_METHOD(ShowPlayerPickerAsync),                        // 11
    ANYXBOXX_METHOD(ShowPlayerPickerResultCount),                  // 12
    ANYXBOXX_METHOD(ShowPlayerPickerResult),                       // 13
    ANYXBOXX_METHOD(ShowErrorDialogAsync),                         // 14
    ANYXBOXX_METHOD(ShowErrorDialogResult),                        // 15
    ANYXBOXX_METHOD(SetNotificationPositionHint),                  // 16
    ANYXBOXX_METHOD(ShowTextEntryAsync),                           // 17
    ANYXBOXX_METHOD(ShowTextEntryResultSize),                      // 18
    ANYXBOXX_METHOD(ShowTextEntryResult),                          // 19
    ANYXBOXX_NOT_IMPLEMENTED,  // 20 (absente de gdk_api_table.inc)
    ANYXBOXX_NOT_IMPLEMENTED,  // 21 (absente de gdk_api_table.inc)
    ANYXBOXX_NOT_IMPLEMENTED,  // 22 (absente de gdk_api_table.inc)
    ANYXBOXX_NOT_IMPLEMENTED,  // 23 (absente de gdk_api_table.inc)
    ANYXBOXX_NOT_IMPLEMENTED,  // 24 XGameUiShowWebAuthenticationAsync
    ANYXBOXX_NOT_IMPLEMENTED,  // 25 XGameUiShowWebAuthenticationResultSize
    ANYXBOXX_NOT_IMPLEMENTED,  // 26 XGameUiShowWebAuthenticationResult
    ANYXBOXX_NOT_IMPLEMENTED,  // 27 XGameUiShowWebAuthenticationWithOptionsAsync
    ANYXBOXX_NOT_IMPLEMENTED,  // 28 XGameUiShowManageSpaceAsync
    ANYXBOXX_NOT_IMPLEMENTED,  // 29 XGameUiShowManageSpaceResult
    ANYXBOXX_METHOD(ShowMultiplayerActivityGameInviteAsync),       // 30
    ANYXBOXX_METHOD(ShowActivityInviteResult),                     // 31 XGameUiShowMultiplayerActivityGameInviteResult
    ANYXBOXX_NOT_IMPLEMENTED,  // 32 (absente de gdk_api_table.inc)
    ANYXBOXX_NOT_IMPLEMENTED,  // 33 (absente de gdk_api_table.inc)
    ANYXBOXX_NOT_IMPLEMENTED,  // 34 XGameUiTextEntryOpen
    ANYXBOXX_NOT_IMPLEMENTED,  // 35 XGameUiTextEntryClose
    ANYXBOXX_NOT_IMPLEMENTED,  // 36 XGameUiTextEntryGetState
    ANYXBOXX_NOT_IMPLEMENTED,  // 37 XGameUiTextEntryGetExtents
    ANYXBOXX_NOT_IMPLEMENTED,  // 38 XGameUiTextEntryUpdatePositionHint
    ANYXBOXX_NOT_IMPLEMENTED,  // 39 XGameUiTextEntryUpdateVisibility
    ANYXBOXX_NOT_IMPLEMENTED,  // 40 XGameUiShowStateShareAsync
    ANYXBOXX_NOT_IMPLEMENTED,  // 41 XGameUiShowStateShareResult
    ANYXBOXX_METHOD(SetUiCallbacks),                               // 42 XGameUiSetUiCallbacks
    ANYXBOXX_METHOD(SetMessageDialogUiResponse),                   // 43
    ANYXBOXX_METHOD(SetPlayerPickerUiResponse),                    // 44
    ANYXBOXX_METHOD(SetTextEntryUiResponse),                       // 45
    ANYXBOXX_METHOD(DismissedResponse<UiKind::PlayerProfileCard>), // 46 XGameUiSetPlayerProfileCardUiResponse
    ANYXBOXX_METHOD(DismissedResponse<UiKind::SendGameInvite>),    // 47 XGameUiSetSendGameInviteUiResponse
    ANYXBOXX_METHOD(DismissedResponse<UiKind::Achievements>),      // 48 XGameUiSetAchievementsUiResponse
    ANYXBOXX_METHOD(DismissedResponse<UiKind::MultiplayerActivityGameInvite>),  // 49
    ANYXBOXX_METHOD(DismissedResponse<UiKind::ErrorDialog>),       // 50 XGameUiSetErrorDialogUiResponse
};

const anyxboxx::StaticInterface kGameUi{kGameUiMethods};

}  // namespace

namespace anyxboxx {

void* GameUiInterface() { return const_cast<StaticInterface*>(&kGameUi); }

}  // namespace anyxboxx
