import cv2
url = 'rtsp://admin:Wurenchuan619@192.168.1.64:554/h264/ch33/main/av_stream'
cap = cv2.VideoCapture(url)
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
