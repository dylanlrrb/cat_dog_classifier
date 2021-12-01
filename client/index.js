
var width = 0;    // We will scale the photo width to this
var height = 0;     // This will be computed based on the input stream

var streaming = false;

var loader = null;
var photo = null
var video = null;
var canvas = null;
var startbutton = null;
var retakebutton = null;
var classifybutton = null;
var resultspane = null;
var tryagainbutton = null;

var certainty = null;
var animal = null;
var speed = null;

function clearphoto() {
  var context = canvas.getContext('2d');
  context.fillStyle = "#AAA";
  context.fillRect(0, 0, canvas.width, canvas.height);

  video.classList.remove("hidden");
  canvas.classList.add("hidden");
  startbutton.classList.remove("hidden");
  retakebutton.classList.add("hidden");
  classifybutton.classList.add("hidden");
  resultspane.classList.add("hidden");

  // scroll to top
  window.scrollTo(0, 0);

  var data = canvas.toDataURL('image/jpeg');
  photo.setAttribute('src', data);
}

function takepicture() {
  var context = canvas.getContext('2d');
  if (width && height) {
    canvas.width = width;
    canvas.height = height;
    context.drawImage(video, 0, 0, width, height);

    video.classList.add("hidden");
    canvas.classList.remove("hidden");
    startbutton.classList.add("hidden");
    retakebutton.classList.remove("hidden");
    classifybutton.classList.remove("hidden");

    var data = canvas.toDataURL('image/png');
    photo.setAttribute('src', data);
  } else {
    clearphoto();
  }
}

function classify() {
  loader.classList.remove("hidden");
  fetch('/classify')
  .then(data => data.json())
  .then(data => {
    console.log(data);
    // unhide results pane
    loader.classList.add("hidden");
    resultspane.classList.remove("hidden");
    // fill out fields
    certainty.innerHTML = data.certainty[0];
    animal.innerHTML = data.mapping[data.ranking[0]];
    speed.innerHTML = data.speed
    // scroll to bottom
    window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
  })
  .catch(e => console.error(e));
}

function startup() {
  body = document.getElementById('body');
  loader = document.getElementById('loader');
  photo = document.getElementById('photo');
  video = document.getElementById('video');
  canvas = document.getElementById('canvas');
  startbutton = document.getElementById('startbutton');
  retakebutton = document.getElementById('retakebutton');
  classifybutton = document.getElementById('classifybutton');
  resultspane = document.getElementById('resultspane');
  tryagainbutton = document.getElementById('tryagainbutton');
  certainty = document.getElementById('certainty');
  animal = document.getElementById('animal');
  speed = document.getElementById('speed');

  navigator.mediaDevices.getUserMedia({ video: {
      facingMode: 'environment'
    }, audio: false })
    .then(function(stream) {
        video.srcObject = stream;
        video.play();
    })
    .catch(function(err) {
        console.log("An error occurred: " + err);
    });
  
    video.addEventListener('canplay', function(ev){
    if (!streaming) {
      width = body.getBoundingClientRect().width * 0.8
      height = video.videoHeight / (video.videoWidth/width);

      video.setAttribute('width', width);
      video.setAttribute('height', height);
      canvas.setAttribute('width', width);
      canvas.setAttribute('height', height);
      streaming = true;

      video.classList.remove("visually-hidden");
      startbutton.classList.remove("visually-hidden");
      loader.classList.add("hidden");
    }
  }, false);
  
  startbutton.addEventListener('click', function(ev){
    takepicture();
    ev.preventDefault();
  }, false);

  retakebutton.addEventListener('click', function(ev){
    clearphoto();
    ev.preventDefault();
  }, false);

  classifybutton.addEventListener('click', function(ev){
    classify();
    ev.preventDefault();
  }, false);

  tryagainbutton.addEventListener('click', function(ev){
    clearphoto();
    ev.preventDefault();
  }, false);

  
  clearphoto();
}

document.addEventListener('DOMContentLoaded', function() {
  console.log('document is ready.');
  startup();
});
