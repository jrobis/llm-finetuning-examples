for i in 0{0..9} {10..29} ; do
    echo "Downloading $i file..."
    wget -P /mnt/share/the_pile "https://the-eye.eu/public/AI/pile/train/$i.jsonl.zst"
done