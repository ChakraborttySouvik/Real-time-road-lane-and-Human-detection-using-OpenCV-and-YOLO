import urllib.request

# Download yolov3.cfg
cfg_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
urllib.request.urlretrieve(cfg_url, "yolov3.cfg")
names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
urllib.request.urlretrieve(names_url, "coco.names")

print("Files downloaded successfully!")