const video = document.getElementById('video-input');
const canvas = document.getElementById('canvas-output');

(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
    });

    let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    let dst = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    let gray = new cv.Mat();
    let cap = new cv.VideoCapture(video);
    let faces = new cv.RectVector();
    let classifier = new cv.CascadeClassifier();
    let utils = new Utils("errorMessage");
    
    classifier.load('haarcascade_frontalface_default.xml');

    video.srcObject = stream
    video.play()

    const FPS = 30;

    function processVideo() {
        let begin = Date.now();

        cap.read(src);
        src.copyTo(dst);
        cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY);

        classifier.detectMultiScale(gray, faces, 1.1, 3, 0);
        
        // for (let i = 0; i < faces.size(); ++i) {
        //     let face = faces.get(i);
        //     let point1 = new cv.Point(face.x, face.y);
        //     let point2 = new cv.Point(face.x + face.width, face.y + face.height);
        //     cv.rectangle(dst, point1, point2, [255, 0, 0, 255]);
        // }

        cv.imshow('canvas-output', gray);

        let delay = 1000/FPS - (Date.now() - begin);
        setTimeout(processVideo, delay);
    }

    setTimeout(processVideo, 0)
})();