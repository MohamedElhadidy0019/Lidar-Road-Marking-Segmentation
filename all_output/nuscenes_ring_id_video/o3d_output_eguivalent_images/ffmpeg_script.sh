ffmpeg -framerate 12 -pattern_type glob -i '*.jpg' \
  -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
