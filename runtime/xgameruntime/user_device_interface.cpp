// XUser et périphériques : point de terminaison audio réel de Windows, manette sur PC.
//
// Méthodes : gdk_api_table.inc (interface {0CC6A956…}). Signatures et comportement sur PC :
// https://learn.microsoft.com/gaming/gdk/docs/reference/system/xuser/functions/<nom en minuscules>
// - XUserGetDefaultAudioEndpointUtf16 : identifiant du périphérique de communication par défaut
//   de Windows (IMMDeviceEnumerator::GetDefaultAudioEndpoint, rôle eCommunications), celui que
//   l'exemple de la page passe ensuite à IMMDeviceEnumerator::GetDevice.
// - XUserFindControllerForUserWithUiAsync : « On desktop, this function will not show UI », et
//   le résultat est « always ... an all-zero device ID ».
// - Abonnements XUserRegisterForDeviceAssociationChanged et
//   XUserRegisterForDefaultAudioEndpointUtf16Changed : E_NOTIMPL, journalisé (règle 3). AnyXboxX ne
//   suit ni les associations manette-joueur ni les changements du périphérique audio par défaut de
//   Windows, il ne pourrait jamais appeler le rappel. XUserGetDefaultAudioEndpointUtf16 rend la
//   valeur courante à chaque appel (docs/progress.md, « Change subscriptions »).
// XUserFindForDevice reste non écrit : la page ne dit pas ce que répond un PC.
#include <windows.h>
#include <mmdeviceapi.h>

#include <cstdint>
#include <cwchar>
#include <utility>

#include "async_call.h"
#include "interface_helpers.h"
#include "interfaces.h"
#include "trace.h"

namespace {

constexpr HRESULT E_GAMEUSER_USER_NOT_FOUND = static_cast<HRESULT>(0x89245104L);
// XUserDefaultAudioEndpointKind (enums/xuserdefaultaudioendpointkind).
enum EndpointKind : uint32_t { kCommunicationRender = 0, kCommunicationCapture = 1 };

// APP_LOCAL_DEVICE_ID : 32 octets (structs/app_local_device_id).
struct LocalDeviceId {
  uint8_t value[32];
};

// XUserLocalId : structure d'un seul uint64_t, passée par valeur.
struct UserLocalId {
  uint64_t value;
};

const char kControllerIdentity = 0;

struct ComInit {
  HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  ~ComInit() {
    if (SUCCEEDED(hr)) CoUninitialize();
  }
};

template <typename T>
struct ComPtr {
  T* raw = nullptr;
  ~ComPtr() {
    if (raw != nullptr) raw->Release();
  }
  T** Out() { return &raw; }
  T* operator->() const { return raw; }
};

// Copie `id` (zéro final compris) dans le tampon du jeu.
HRESULT CopyEndpointId(const wchar_t* id, size_t count, wchar_t* buffer, size_t* used) {
  const size_t needed = std::wcslen(id) + 1;
  if (used != nullptr) *used = needed;
  if (buffer == nullptr || count < needed) return HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER);
  std::wmemcpy(buffer, id, needed);
  return S_OK;
}

HRESULT XUserGetDefaultAudioEndpointUtf16(UserLocalId user, uint32_t kind, size_t count, wchar_t* buffer,
                                          size_t* used) noexcept {
  if (!anyxboxx::IsLocalUserLocalId(user.value)) return E_GAMEUSER_USER_NOT_FOUND;
  if (kind != kCommunicationRender && kind != kCommunicationCapture) return E_INVALIDARG;
  const ComInit com;
  ComPtr<IMMDeviceEnumerator> enumerator;
  HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator),
                                reinterpret_cast<void**>(enumerator.Out()));
  if (FAILED(hr)) return hr;
  ComPtr<IMMDevice> device;
  hr = enumerator->GetDefaultAudioEndpoint(kind == kCommunicationRender ? eRender : eCapture, eCommunications,
                                           device.Out());
  if (FAILED(hr)) return hr;  // E_NOTFOUND sans périphérique, comme l'attend l'exemple de la page
  wchar_t* id = nullptr;
  hr = device->GetId(&id);
  if (FAILED(hr)) return hr;
  hr = CopyEndpointId(id, count, buffer, used);
  CoTaskMemFree(id);
  ANYXBOXX_TRACE("[runtime] XUserGetDefaultAudioEndpointUtf16(%u) : 0x%08lX", kind, static_cast<unsigned long>(hr));
  return hr;
}

HRESULT XUserFindControllerForUserWithUiAsync(void* user, XAsyncBlock* async) noexcept {
  uint64_t xuid = 0;
  if (!anyxboxx::LocalUserId(user, &xuid)) return E_INVALIDARG;
  anyxboxx::AsyncCall call;
  call.size = sizeof(LocalDeviceId);
  call.write = [](void* buffer) { *static_cast<LocalDeviceId*>(buffer) = LocalDeviceId{}; };
  return anyxboxx::BeginAsyncCall(async, &kControllerIdentity, "XUserFindControllerForUserWithUiAsync",
                                  std::move(call));
}

HRESULT XUserFindControllerForUserWithUiResult(XAsyncBlock* async, LocalDeviceId* device) noexcept {
  return XAsyncGetResult(async, &kControllerIdentity, sizeof *device, device, nullptr);
}

// Désabonnements (jeton, attendre) : aucune inscription ne peut réussir (E_NOTIMPL), rien à retirer.
bool Unsubscribe(uint64_t, bool) noexcept { return false; }

// Méthodes 3 à 10 de l'interface {0CC6A956…} (gdk_api_table.inc).
void* const kUserDeviceMethods[] = {
    ANYXBOXX_IUNKNOWN_METHODS,                                // 0-2
    ANYXBOXX_NOT_IMPLEMENTED,  // 3 XUserFindForDevice
    ANYXBOXX_NOT_IMPLEMENTED,  // 4 XUserRegisterForDeviceAssociationChanged
    ANYXBOXX_METHOD(Unsubscribe),                             // 5 XUserUnregisterForDeviceAssociationChanged
    ANYXBOXX_METHOD(XUserGetDefaultAudioEndpointUtf16),       // 6
    ANYXBOXX_NOT_IMPLEMENTED,  // 7 XUserRegisterForDefaultAudioEndpointUtf16Changed
    ANYXBOXX_METHOD(Unsubscribe),                             // 8 XUserUnregisterForDefaultAudioEndpointUtf16Changed
    ANYXBOXX_METHOD(XUserFindControllerForUserWithUiAsync),   // 9
    ANYXBOXX_METHOD(XUserFindControllerForUserWithUiResult),  // 10
};

const anyxboxx::StaticInterface kUserDevice{kUserDeviceMethods};

}  // namespace

namespace anyxboxx {

void* UserDeviceInterface() { return const_cast<StaticInterface*>(&kUserDevice); }

}  // namespace anyxboxx
