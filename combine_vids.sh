
ffmpeg -i $1.mp4 -s 1280x720 -c:a copy $1.mp4
ffmpeg -i $2.mp4 -s 1280x720 -c:a copy $2.mp4
# ffmpeg -i $1.mp4 -i $2.mp4 -filter_complex hstack output.mp4
# ffmpeg -i left.mp4 -i right.mp4 -filter_complex hstack output.mp4

# ffmpeg -i $1.mp4 -i $2.mp4 -filter_complex "[0:v]crop=360:360:0:140,fps=5[v0];[1:v]fps=30[v1];[v0][v1]vstack=inputs=2,split=3[lc][m][rc];[lc]crop=101:ih:0:0[l];[rc]crop=101:ih:259:0[r];[l][m][r]hstack=inputs=3[v];[0:a][1:a]amix[a]" -map "[v]" -map "[a]" -preset ultrafast ./stackedOutput.mp4


ffmpeg \
  -i $1.mp4 \
  -i $2.mp4 \
  -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/1:0[vid]' \
  -map '[vid]' \
  -c:v libx264 \
  -crf 23 \
  -preset veryfast \
  output.mp4

mpv output.mp4
