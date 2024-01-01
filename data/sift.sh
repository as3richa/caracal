
#!/bin/bash

set -euxo pipefail

cd data

fetch_archive() {
    local url="$1"
    local md5_checksum="$2"

    local tmp_filename="tmp.tar.gz"

    wget $url -O "$tmp_filename"
    local observed_md5_checksum="$(md5sum $tmp_filename | cut -d' ' -f1)"

    if [ "$observed_md5_checksum" != "$md5_checksum" ]; then
        echo "Checksum mismatch on $url"
        rm "$tmp_filename"
        exit 1
    fi

    tar -xvf "$tmp_filename"
    rm "$tmp_filename"
}

fetch_archive "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz" "0b8324a7a82d7f2663d7dcbd57642df7"
fetch_archive "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz" "b23d1b3b2ee8469d819b61ca900ef0ed"