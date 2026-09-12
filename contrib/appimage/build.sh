#!/usr/bin/env bash
# Builds pc4l-gui-<version>-x86_64.AppImage from a release build.
#
#   contrib/appimage/build.sh [OUT_DIR]
#
# Needs target/release/pc4l-gui and pc4l (cargo build --release), the
# libwebgpu_dawn.so beside them, and `appimagetool` on PATH (or APPIMAGETOOL set).
# The AppImage carries the binaries and the WebGPU provider; not the models, which
# it looks for in a `models/` directory next to the AppImage (extract the release's
# pc4l-models-*.tar.gz there) before falling back to a download into ~/.cache.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:-"$root/dist"}
tool=${APPIMAGETOOL:-appimagetool}
version=$(sed -n 's/^version = "\(.*\)"/\1/p' "$root/Cargo.toml" | head -1)

bin="$root/target/release"
for f in pc4l-gui pc4l libwebgpu_dawn.so; do
  [ -e "$bin/$f" ] || { echo "missing $bin/$f (build with cargo build --release)" >&2; exit 1; }
done

appdir=$(mktemp -d)/pc4l-gui.AppDir
mkdir -p "$appdir/usr/bin" "$appdir/usr/share/applications" \
  "$appdir/usr/share/icons/hicolor/scalable/apps"
# -L: the provider is a symlink into ~/.cache/dfbin in a dev build.
cp -L "$bin/pc4l-gui" "$bin/pc4l" "$bin/libwebgpu_dawn.so" "$appdir/usr/bin/"
cp "$here/pc4l-gui.desktop" "$appdir/usr/share/applications/"
cp "$here/pc4l-gui.svg" "$appdir/usr/share/icons/hicolor/scalable/apps/"
ln -s usr/share/applications/pc4l-gui.desktop "$appdir/pc4l-gui.desktop"
ln -s usr/share/icons/hicolor/scalable/apps/pc4l-gui.svg "$appdir/pc4l-gui.svg"
ln -s pc4l-gui.svg "$appdir/.DirIcon"
cat > "$appdir/AppRun" <<'RUN'
#!/bin/sh
# The binaries find libwebgpu_dawn.so beside themselves ($ORIGIN rpath); the
# models are looked for next to the AppImage itself (see local/models.rs).
here=$(dirname "$(readlink -f "$0")")
exec "$here/usr/bin/pc4l-gui" "$@"
RUN
chmod +x "$appdir/AppRun"

mkdir -p "$out"
ARCH=x86_64 "$tool" --no-appstream "$appdir" "$out/pc4l-gui-v$version-x86_64.AppImage"
rm -rf "$(dirname "$appdir")"
echo "$out/pc4l-gui-v$version-x86_64.AppImage"
