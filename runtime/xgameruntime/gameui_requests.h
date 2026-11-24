// XGameUi : demandes d'interface confiées aux rappels du jeu (XGameUiSetUiCallbacks).
//
// Sans Gaming Services, il n'y a pas d'interface système. Le GDK documente une autre voie :
// le jeu fournit ses propres écrans par XGameUiSetUiCallbacks, reçoit chaque demande avec un
// XGameUiCallbackHandle, puis donne la réponse de son écran par XGameUiSet*UiResponse.
// https://learn.microsoft.com/gaming/gdk/docs/reference/system/xgameui/structs/xgameuiuicallbacks
#pragma once

#include <windows.h>

#include <cstdint>
#include <functional>
#include <vector>

#include "XAsync.h"

namespace anyxboxx {

// Ordre des rappels de la structure XGameUiUiCallbacks (après `context`).
enum class UiKind : uint32_t {
  PlayerProfileCard,
  PlayerPicker,
  SendGameInvite,
  Achievements,
  MultiplayerActivityGameInvite,
  MessageDialog,
  ErrorDialog,
  TextEntry,
  Count,
};

// XGameUiUiCallbacks : contexte du jeu puis un rappel par écran (nul : interface système).
struct UiCallbacks {
  void* context;
  void* callbacks[static_cast<uint32_t>(UiKind::Count)];
};

// XGameUiSetUiCallbacks : remplace tous les rappels (nullptr les retire).
void SetUiCallbacks(const UiCallbacks* callbacks);

// Rappel du jeu pour `kind` et son contexte ; faux si le jeu n'en a pas fourni.
bool FindUiCallback(UiKind kind, void** callback, void** context);

// Appelle le rappel du jeu avec (poignée de la demande, file de l'opération).
using UiInvoke = std::function<void(void* handle, void* queue)>;

// Lance la demande : `invoke` part sur la file de `async`, et l'opération reste en cours
// jusqu'à la réponse du jeu (CompleteUiRequest) ou jusqu'à son annulation (E_ABORT).
HRESULT BeginUiRequest(XAsyncBlock* async, const void* identity, const char* name, UiKind kind, UiInvoke invoke);

// Réponse du jeu : E_INVALIDARG si la poignée est inconnue, déjà répondue, annulée, ou
// d'un autre écran. `result` devient le résultat lu par les fonctions *Result.
HRESULT CompleteUiRequest(void* handle, UiKind kind, std::vector<uint8_t> result);

}  // namespace anyxboxx
