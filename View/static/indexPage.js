document.addEventListener('DOMContentLoaded', function () {

    function isValidFile(file) {
        const validTypes = ['image/jpeg', 'image/png', 'image/gif'];
        const maxSize = 2 * 1024 * 1024; // 2MB
        if (!validTypes.includes(file.type)) {
            alert("Only JPG, PNG, and GIF files are allowed.");
            return false;
        }
        if (file.size > maxSize) {
            alert("The file is too large. Please upload files under 2MB.");
            return false;
        }
        return true;
    }


    function handleFile(file) {

        // Usage within handleFile
        if (!isValidFile(file)) {
            return; // Stop processing the file if it's invalid
        }

        let reader = new FileReader();
        reader.onload = function (e) {
            console.log(reader.result);
            document.getElementById('profile').style.backgroundImage = 'url(' + e.target.result + ')';
            document.getElementById('profile').classList.add('hasImage');
        };
        reader.onerror = function (e) {
            console.error("Error reading file:", e);
        };
        reader.readAsDataURL(file);
    }

    let profile = document.getElementById('profile');

    profile.addEventListener('dragover', function () {
        this.classList.add('dragging');
    });

    profile.addEventListener('dragleave', function () {
        this.classList.remove('dragging');
    });

    profile.addEventListener('drop', function (e) {
        e.preventDefault();
        let files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
        this.classList.remove('dragging');
    });

    document.getElementById('mediaFile').addEventListener('change', function (e) {
        let files = e.target.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    profile.addEventListener('click', function () {
        document.getElementById('mediaFile').click();
    });

    window.addEventListener('dragover', function (e) {
        e.preventDefault();
    });

    window.addEventListener('drop', function (e) {
        e.preventDefault();
    });


    const uploadBtn = document.getElementById('uploadBtn');
    const fileInput = document.getElementById('mediaFile');
    const profileDiv = document.getElementById('profile');

    uploadBtn.addEventListener('click', async function () {
        let file = fileInput.files[0];

        // Check if a file is selected
        if (!file) {
            alert('Please select a file to upload.');
            return;
        }

        // Prepare FormData
        let formData = new FormData();
        formData.append('file', file);

        // Disable the button during upload
        uploadBtn.disabled = true;
        uploadBtn.textContent = 'Processing...';

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData,
            });
            // Assuming the response contains the URL to redirect to
            const {redirectTo} = await response.json(); // Server sends JSON response with redirectTo field

            // Redirect to the result.html template
            window.location.href = redirectTo; // Redirects the user to the specified URL
        } catch (error) {
            console.error('Error:', error);
            alert('Upload failed.');
        } finally {
            // Re-enable the button after upload
            uploadBtn.disabled = false;
            uploadBtn.textContent = 'Upload';
        }
    });
});


// init
var maxx = document.body.clientWidth;
var maxy = document.body.clientHeight;
var halfx = maxx / 2;
var halfy = maxy / 2;
var canvas = document.createElement("canvas");
document.body.appendChild(canvas);
canvas.width = maxx;
canvas.height = maxy;
var context = canvas.getContext("2d");
var dotCount = 200;
var dots = [];
// create dots
for (var i = 0; i < dotCount; i++) {
  dots.push(new dot());
}

// dots animation
function render() {
  context.fillStyle = "#000000";
  context.fillRect(0, 0, maxx, maxy);
  for (var i = 0; i < dotCount; i++) {
    dots[i].draw();
    dots[i].move();
  }
  requestAnimationFrame(render);
}

// dots class
// @constructor
function dot() {

  this.rad_x = 2 * Math.random() * halfx + 1;
  this.rad_y = 1.2 * Math.random() * halfy + 1;
  this.alpha = Math.random() * 360 + 1;
  this.speed = Math.random() * 100 < 50 ? 1 : -1;
  this.speed *= 0.1;
  this.size = Math.random() * 5 + 1;
  this.color = Math.floor(Math.random() * 256);

}

// drawing dot
dot.prototype.draw = function() {

  // calc polar coord to decart
  var dx = halfx + this.rad_x * Math.cos(this.alpha / 180 * Math.PI);
  var dy = halfy + this.rad_y * Math.sin(this.alpha / 180 * Math.PI);
  // set color
  context.fillStyle = "rgb(" + this.color + "," + this.color + "," + this.color + ")";
  // draw dot
  context.fillRect(dx, dy, this.size, this.size);

};

// calc new position in polar coord
dot.prototype.move = function() {

  this.alpha += this.speed;
  // change color
  if (Math.random() * 100 < 50) {
    this.color += 1;
  } else {
    this.color -= 1;
  }

};

// start animation
render();