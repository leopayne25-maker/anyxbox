#!/bin/bash
# Build a GE-Proton wineserver with the view-tree patch (CachyOS/wine-cachyos#50, Wine bug 60868) and make a copy
# of that GE-Proton with only files/bin/wineserver replaced.
#
#   scripts/build_ge_wineserver.sh <GE tag, e.g. GE-Proton11-14> <installed GE-Proton dir> <work dir>
#
# Steps: clone proton-ge-custom at the tag, fetch its wine and wine-staging submodules at the pinned commits, run the
# wine section of GE's patches/protonprep-valve-staging.sh, generate the headers configure needs, apply the patch,
# build only server/wineserver, check its protocol version against the installed Proton's ntdll.so, copy the Proton.
set -euo pipefail
TAG=${1:?GE tag}; INST=$(realpath "${2:?installed GE-Proton dir}"); WORK=${3:?work dir}
PATCH_URL=https://github.com/CachyOS/wine-cachyos/commit/fd66dbca8eb0754839ca974e45b5ea446ca62d2f.patch
mkdir -p "$WORK"; WORK=$(realpath "$WORK"); cd "$WORK"
[ -e src ] && { echo "$WORK/src exists: use an empty work dir"; exit 1; }

git clone -q --depth 1 -b "$TAG" https://github.com/GloriousEggroll/proton-ge-custom src
cd src
for m in wine wine-staging; do
  url=$(git config -f .gitmodules "submodule.$m.url"); sha=$(git ls-tree HEAD "$m" | awk '{print $3}')
  rm -rf "$m"; git init -q "$m"; git -C "$m" fetch -q --depth 1 "$url" "$sha"; git -C "$m" checkout -q FETCH_HEAD
  echo "$m $url $sha"
done
P=patches/protonprep-valve-staging.sh
s=$(grep -n '^### (2) WINE PATCHING ###' $P | cut -d: -f1); e=$(grep -n '^### END WINE PATCHING ###' $P | cut -d: -f1)
{ sed -n '1,/^### (1) PREP SECTION ###/p' $P | head -n -1; sed -n "${s},${e}p" $P; } > ../ge-prep-wine.sh
bash ../ge-prep-wine.sh > ../prep.log 2>&1
cd wine
XDG_CACHE_HOME="$WORK/cache" dlls/winevulkan/make_vulkan -x vk.xml -X video.xml > ../../vk.log 2>&1
tools/make_specfiles > ../../spec.log 2>&1
curl -fsSL "$PATCH_URL" -o ../../viewtree.patch
git apply ../../viewtree.patch
V=$(sed -n 's/^#define SERVER_PROTOCOL_VERSION //p' include/wine/server_protocol.h)
n=$(objdump -d --no-show-raw-insn "$INST/files/lib/wine/x86_64-unix/ntdll.so" | grep -cE "(cmp|mov) +\\\$0x$(printf %x "$V"),") || true
[ "$n" -ge 1 ] || { echo "protocol $V not found in $INST ntdll.so: wrong source for this Proton"; exit 1; }
echo "protocol $V matches $INST ($n hits)"

mkdir -p "$WORK/build"; cd "$WORK/build"
../src/wine/configure --enable-win64 --disable-tests --without-x --without-freetype --without-gnutls --without-vulkan \
  --without-opengl --without-alsa --without-pulse --without-gstreamer --without-sdl --without-udev --without-usb \
  --without-v4l2 --without-cups --without-dbus --without-krb5 --without-netapi --without-pcap --without-sane \
  --without-unwind --without-wayland --without-xinerama --without-xrandr --without-xinput2 --without-xcomposite \
  --without-xcursor --without-xfixes --without-xshape --without-xshm --without-osmesa --without-fontconfig \
  --without-gphoto --without-ffmpeg --without-capi --without-pcsclite --without-coreaudio --without-oss \
  --without-mingw > configure.log 2>&1
make -j"$(nproc)" server/wineserver > make.log 2>&1
strip -o wineserver server/wineserver
OUT="$WORK/$(basename "$INST")-viewtree"
cp -a "$INST" "$OUT"; cp wineserver "$OUT/files/bin/wineserver"
echo "done: $OUT (only files/bin/wineserver replaced); use PROTON=$OUT/proton"
