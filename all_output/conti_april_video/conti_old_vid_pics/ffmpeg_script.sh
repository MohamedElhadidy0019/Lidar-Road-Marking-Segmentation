ffmpeg -framerate 5 -pattern_type glob -i '*.jpeg' \
  -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
