// XGameStreaming hors ligne : le jeu tourne sur le PC, aucun appareil ne le reçoit en streaming.
//
// Méthodes : gdk_api_table.inc. Signatures, codes d'erreur et comportement sans client :
// https://learn.microsoft.com/gaming/gdk/docs/reference/system/xgamestreaming/xgamestreaming_members
// (une page par fonction sous .../xgamestreaming/functions/<nom en minuscules>).
// Sous AnyXboxX, aucun client de streaming ne se connecte jamais : le nombre de clients vaut 0,
// toute question sur un client précis répond E_GAMESTREAMING_CLIENT_NOT_CONNECTED, et une manette
// n'est jamais une manette de streaming. Rien n'est simulé : aucune réponse ne prétend un stream.
#include <windows.h>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "interface_helpers.h"
#include "interfaces.h"
#include "trace.h"

namespace {

// Codes documentés (page « XGameStreaming (API contents) », section Error Codes).
constexpr HRESULT kNotInitialized = static_cast<HRESULT>(0x89245400);         // E_GAMESTREAMING_NOT_INITIALIZED
constexpr HRESULT kClientNotConnected = static_cast<HRESULT>(0x89245401);     // E_GAMESTREAMING_CLIENT_NOT_CONNECTED
constexpr HRESULT kNoData = static_cast<HRESULT>(0x89245402);                 // E_GAMESTREAMING_NO_DATA
constexpr HRESULT kNoDatacenter = static_cast<HRESULT>(0x89245403);           // E_GAMESTREAMING_NO_DATACENTER
constexpr HRESULT kNotStreamingController = static_cast<HRESULT>(0x89245404); // E_GAMESTREAMING_NOT_STREAMING_CONTROLLER

// XGameStreamingClientId : identifiant opaque de 8 octets, passé par valeur.
using ClientId = uint64_t;
// XGameStreamingConnectionState (enums/xgamestreamingconnectionstate) : Disconnected = 0.
constexpr uint32_t kDisconnected = 0;
// La largeur 16/9 sert de borne aux deux rapports de XGameStreamingGetDisplayDetails.
constexpr float kSixteenByNine = 16.0f / 9.0f;

// XTaskQueueRegistrationToken : structure d'un seul uint64_t, passée par valeur.
struct RegistrationToken {
  uint64_t token;
};

// XGameStreamingInitialize / Uninitialize : les autres API renvoient
// E_GAMESTREAMING_NOT_INITIALIZED tant que l'initialisation n'a pas eu lieu.
std::atomic<bool> g_initialized{false};

bool Initialized() { return g_initialized.load(std::memory_order_acquire); }

// Réponse commune des API qui interrogent un client : aucun n'est connecté.
HRESULT NoClient(const char* function) {
  if (!Initialized()) return kNotInitialized;
  ANYXBOXX_TRACE("[runtime] %s : aucun client de streaming connecté", function);
  return kClientNotConnected;
}

// --- Initialisation (functions/xgamestreaminginitialize, xgamestreaminguninitialize) -----
HRESULT XGameStreamingInitialize() noexcept {
  g_initialized.store(true, std::memory_order_release);
  ANYXBOXX_TRACE("[runtime] XGameStreamingInitialize : jeu local, aucun client de streaming");
  return S_OK;
}
void XGameStreamingUninitialize() noexcept { g_initialized.store(false, std::memory_order_release); }

// --- État de connexion (getclientcount, getclients, getconnectionstate, isstreaming) ------
bool XGameStreamingIsStreaming() noexcept { return false; }
uint32_t XGameStreamingGetClientCount() noexcept { return 0; }

HRESULT XGameStreamingGetClients(uint32_t, ClientId*, uint32_t* clients_used) noexcept {
  if (!Initialized()) return kNotInitialized;
  if (clients_used == nullptr) return E_INVALIDARG;
  *clients_used = 0;
  return S_OK;
}

uint32_t XGameStreamingGetConnectionState(ClientId) noexcept { return kDisconnected; }

// À l'inscription, le rappel part pour chaque client déjà connecté : aucun ici, et aucun
// ne se connectera. Le jeton reste valable jusqu'au désabonnement.
HRESULT RegisterConnectionStateChanged(void*, void*, void* callback, void* token) noexcept {
  if (!Initialized()) return kNotInitialized;
  return anyxboxx::AddSubscription(callback, token);
}
bool UnregisterConnectionStateChanged(RegistrationToken token, bool) noexcept {
  return anyxboxx::RemoveSubscription(token.token);
}

// --- Propriétés d'un client : aucun client, donc E_GAMESTREAMING_CLIENT_NOT_CONNECTED -----
HRESULT RegisterClientPropertiesChanged(ClientId, void*, void*, void*, void*) noexcept {
  return NoClient("XGameStreamingRegisterClientPropertiesChanged");
}
// Aucune inscription n'a pu réussir : il n'y a rien à retirer.
bool UnregisterClientPropertiesChanged(ClientId, RegistrationToken, bool) noexcept { return false; }

HRESULT GetStreamPhysicalDimensions(ClientId, uint32_t*, uint32_t*) noexcept {
  return NoClient("XGameStreamingGetStreamPhysicalDimensions");
}
HRESULT IsTouchInputEnabled(ClientId, bool*) noexcept { return NoClient("XGameStreamingIsTouchInputEnabled"); }
HRESULT GetLastFrameDisplayed(ClientId, void*) noexcept { return NoClient("XGameStreamingGetLastFrameDisplayed"); }
HRESULT GetTouchBundleVersion(ClientId, void*, size_t, char*) noexcept {
  return NoClient("XGameStreamingGetTouchBundleVersion");
}
HRESULT GetClientIPAddress(ClientId, size_t, char*) noexcept { return NoClient("XGameStreamingGetClientIPAddress"); }
HRESULT GetSessionId(ClientId, size_t, char*, size_t*) noexcept { return NoClient("XGameStreamingGetSessionId"); }
// « If no version name is specified in the touch adaptation bundle then the size will be 0. »
size_t GetTouchBundleVersionNameSize(ClientId) noexcept { return 0; }

// Paramètres validés comme le documente la page : maxSupportedPixels > 0,
// widestSupportedAspectRatio >= 16/9 et fini, 0 < tallestSupportedAspectRatio <= 16/9.
HRESULT GetDisplayDetails(ClientId, uint32_t max_pixels, float widest, float tallest, void*) noexcept {
  if (!Initialized()) return kNotInitialized;
  const bool valid = max_pixels > 0 && std::isfinite(widest) && widest >= kSixteenByNine && tallest > 0.0f &&
                     tallest <= kSixteenByNine;
  if (!valid) return E_INVALIDARG;
  return NoClient("XGameStreamingGetDisplayDetails");
}

// La latence documentée ne connaît pas CLIENT_NOT_CONNECTED : sans stream, pas de mesure.
HRESULT GetStreamAddedLatency(ClientId, uint32_t*, uint32_t*, uint32_t*) noexcept {
  return Initialized() ? kNoData : kNotInitialized;
}

// --- Centre de données : le jeu tourne sur le PC, pas dans un centre Azure ---------------
size_t XGameStreamingGetServerLocationNameSize() noexcept { return 0; }
HRESULT XGameStreamingGetServerLocationName(size_t, char*) noexcept {
  return Initialized() ? kNoDatacenter : kNotInitialized;
}

// --- Manette : une lecture GameInput locale ne vient jamais d'une manette de streaming ----
HRESULT ReadingNotFromStream(void*, void*) noexcept {
  return Initialized() ? kNotStreamingController : kNotInitialized;
}

// --- Commandes tactiles : « If the player is not streaming the game ... this function does
// nothing. » Les variantes globales s'adressent à tous les clients connectés, soit aucun.
void XGameStreamingHideTouchControls() noexcept {}
void XGameStreamingShowTouchControlLayout(const char*) noexcept {}
void HideTouchControlsOnClient(ClientId) noexcept {}
void ShowTouchControlLayoutOnClient(ClientId, const char*) noexcept {}

HRESULT UpdateTouchControlsState(size_t count, const void* operations) noexcept {
  if (!Initialized()) return kNotInitialized;
  return count > 0 && operations == nullptr ? E_INVALIDARG : S_OK;
}
HRESULT ShowTouchControlsWithStateUpdate(const char*, size_t count, const void* operations) noexcept {
  return UpdateTouchControlsState(count, operations);
}
HRESULT UpdateTouchControlsStateOnClient(ClientId, size_t, const void*) noexcept {
  return NoClient("XGameStreamingUpdateTouchControlsStateOnClient");
}
HRESULT ShowTouchControlsWithStateUpdateOnClient(ClientId, const char*, size_t, const void*) noexcept {
  return NoClient("XGameStreamingShowTouchControlsWithStateUpdateOnClient");
}

// Méthodes 3 à 34 de l'interface {57786622…} (gdk_api_table.inc).
void* const kStreamingMethods[] = {
    ANYXBOXX_IUNKNOWN_METHODS,                                     // 0-2
    ANYXBOXX_METHOD(XGameStreamingInitialize),                     // 3
    ANYXBOXX_METHOD(XGameStreamingUninitialize),                   // 4
    ANYXBOXX_METHOD(XGameStreamingIsStreaming),                    // 5
    ANYXBOXX_METHOD(RegisterClientPropertiesChanged),              // 6
    ANYXBOXX_METHOD(UnregisterClientPropertiesChanged),            // 7
    ANYXBOXX_METHOD(GetStreamPhysicalDimensions),                  // 8
    ANYXBOXX_METHOD(XGameStreamingGetClientCount),                 // 9
    ANYXBOXX_METHOD(XGameStreamingGetClients),                     // 10
    ANYXBOXX_METHOD(XGameStreamingGetConnectionState),             // 11
    ANYXBOXX_METHOD(RegisterConnectionStateChanged),               // 12
    ANYXBOXX_METHOD(UnregisterConnectionStateChanged),             // 13
    ANYXBOXX_METHOD(GetStreamAddedLatency),                        // 14
    ANYXBOXX_METHOD(XGameStreamingGetServerLocationNameSize),      // 15
    ANYXBOXX_METHOD(XGameStreamingGetServerLocationName),          // 16
    ANYXBOXX_METHOD(XGameStreamingHideTouchControls),              // 17
    ANYXBOXX_METHOD(XGameStreamingShowTouchControlLayout),         // 18
    ANYXBOXX_METHOD(HideTouchControlsOnClient),                    // 19
    ANYXBOXX_METHOD(ShowTouchControlLayoutOnClient),               // 20
    ANYXBOXX_METHOD(IsTouchInputEnabled),                          // 21
    ANYXBOXX_METHOD(GetLastFrameDisplayed),                        // 22
    ANYXBOXX_METHOD(ReadingNotFromStream),                         // 23 XGameStreamingGetAssociatedFrame
    ANYXBOXX_METHOD(ReadingNotFromStream),                         // 24 XGameStreamingGetGamepadPhysicality
    ANYXBOXX_METHOD(UpdateTouchControlsState),                     // 25
    ANYXBOXX_METHOD(UpdateTouchControlsStateOnClient),             // 26
    ANYXBOXX_METHOD(ShowTouchControlsWithStateUpdate),             // 27
    ANYXBOXX_METHOD(ShowTouchControlsWithStateUpdateOnClient),     // 28
    ANYXBOXX_METHOD(GetTouchBundleVersionNameSize),                // 29
    ANYXBOXX_METHOD(GetTouchBundleVersion),                        // 30
    ANYXBOXX_METHOD(GetClientIPAddress),                           // 31
    ANYXBOXX_METHOD(GetSessionId),                                 // 32
    ANYXBOXX_METHOD(GetDisplayDetails),                            // 33
    // La résolution du stream s'applique aux clients connectés ; la page ne dit pas ce que
    // répond une console qui ne diffuse pas : non écrite plutôt qu'un succès supposé.
    ANYXBOXX_NOT_IMPLEMENTED,  // 34 XGameStreamingSetResolution
};

const anyxboxx::StaticInterface kStreaming{kStreamingMethods};

}  // namespace

namespace anyxboxx {

void* StreamingInterface() { return const_cast<StaticInterface*>(&kStreaming); }

}  // namespace anyxboxx
