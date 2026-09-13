#!/usr/bin/env bash
# Builds squigl-<version>-x86_64.AppImage from a release build.
#
#   contrib/appimage/build.sh [OUT_DIR]
#
# Needs target/release/squigl and squigl-cli (cargo build --release), the
# libwebgpu_dawn.so beside them, and `appimagetool` on PATH (or APPIMAGETOOL set).
# The AppImage carries the binaries and the WebGPU provider; not the models, which
# it looks for in a `models/` directory next to the AppImage (extract the release's
# squigl-models-*.tar.gz there) before falling back to a download into ~/.cache.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:-"$root/dist"}
tool=${APPIMAGETOOL:-appimagetool}
version=$(sed -n 's/^version = "\(.*\)"/\1/p' "$root/Cargo.toml" | head -1)

bin="$root/target/release"
for f in squigl squigl-cli libwebgpu_dawn.so; do
  [ -e "$bin/$f" ] || { echo "missing $bin/$f (build with cargo build --release)" >&2; exit 1; }
done

appdir=$(mktemp -d)/squigl.AppDir
mkdir -p "$appdir/usr/bin" "$appdir/usr/share/applications" \
  "$appdir/usr/share/icons/hicolor/scalable/apps"
# -L: the provider is a symlink into ~/.cache/dfbin in a dev build.
cp -L "$bin/squigl" "$bin/squigl-cli" "$bin/libwebgpu_dawn.so" "$appdir/usr/bin/"
cp "$here/squigl.desktop" "$appdir/usr/share/applications/"
cp "$here/squigl.svg" "$appdir/usr/share/icons/hicolor/scalable/apps/"
ln -s usr/share/applications/squigl.desktop "$appdir/squigl.desktop"
ln -s usr/share/icons/hicolor/scalable/apps/squigl.svg "$appdir/squigl.svg"
ln -s squigl.svg "$appdir/.DirIcon"
cat > "$appdir/AppRun" <<'RUN'
#!/bin/sh
# The binaries find libwebgpu_dawn.so beside themselves ($ORIGIN rpath); the
# models are looked for next to the AppImage itself (see local/models.rs).
here=$(dirname "$(readlink -f "$0")")
exec "$here/usr/bin/squigl" "$@"
RUN
chmod +x "$appdir/AppRun"

mkdir -p "$out"
ARCH=x86_64 "$tool" --no-appstream "$appdir" "$out/squigl-v$version-x86_64.AppImage"
rm -rf "$(dirname "$appdir")"
echo "$out/squigl-v$version-x86_64.AppImage"
