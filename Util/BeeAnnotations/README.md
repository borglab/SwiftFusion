# How to annotate a video

## Step 1: Save frames as jpegs.

You should save the frames as jpeg, numbered consecutively `frame0.jpeg`,
`frame1.jpeg`, etc.

`youtube-dl` is useful for downloading videos from YouTube.

Here's a command that turns a video files into frame jpegs.
```bash
ffmpeg -ss 163 -t 4 -i bee_video_1.mkv -start_number 0 bee_video_1/frame%d.jpeg
```

`-ss` specifies the start time (seconds) of the interval that frames are taken
from. `-t` specifies the duration (seconds) of the interval that frames are
taken from.

## Step 2: Run the annotation tool.

Update `frameDir` in `app.js` to point at the directory containing the frames.
Update `frameWidth` and `frameHeight` to be the width and height of the frames
in pixels.

Open `annotator.html` in a browser. Usage:
* Click on the frame to move the bounding box.
* W/A/S/D keys also move the bounding box.
* Q/E keys rotate the bounding box.
* The text fields adjust the x, y, theta, width, and height of the bounding box.
* Left/Right arrow keys change the frame.

When you are done, save the contents of the text area to a file named
`track0.txt` (or `track1.txt`, `track2.txt`) in the same directory as the
frames.

Now you can load the frames and the tracks from Swift using:

```swift
BeeVideo(directory: URL(fileURLWithPath: "<dir containing frames and tracks>"))
```
