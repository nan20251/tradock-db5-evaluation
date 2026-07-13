#!/bin/bash
# Download the official HDOCK createpl_linux helper.
# This does not download HDOCKlite itself; HDOCKlite requires the official form.

set -e

TOOLS_DIR="${TOOLS_DIR:-/root/autodl-tmp/tools/hdocklite}"
URL="${CREATEPL_URL:-http://hdock.phys.hust.edu.cn/createpl_linux}"

mkdir -p "$TOOLS_DIR"
curl -L "$URL" -o "$TOOLS_DIR/createpl_linux"
chmod +x "$TOOLS_DIR/createpl_linux"

echo "$TOOLS_DIR/createpl_linux"
