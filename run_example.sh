#!/bin/bash
mkdir -p data/input
mkdir -p results

echo "Generating sample images..."

cat <<EOF > data/input/sample1.ppm
P3
4 4
255
255 0 0  255 0 0  255 0 0  255 0 0
255 0 0  0 255 0  0 255 0  255 0 0
255 0 0  0 255 0  0 255 0  255 0 0
255 0 0  255 0 0  255 0 0  255 0 0
EOF

cat <<EOF > data/input/sample2.ppm
P3
4 4
255
0 0 255  0 0 255  0 0 255  0 0 255
0 255 255  0 255 255  0 255 255  0 0 255
0 255 255  0 255 255  0 255 255  0 0 255
0 0 255  0 0 255  0 0 255  0 0 255
EOF

echo "--- Running blur ---"
./echoedge_gpu.exe -k blur -i data/input -o results

echo "--- Running sobel ---"
./echoedge_gpu.exe -k sobel -i data/input -o results

echo "Done! Check results/ folder."
