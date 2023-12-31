function submitForm() {
    document.getElementById('imageForm').submit();
}

const currentUrl = window.location.href;
console.log(currentUrl);

// POST data to the server
function submitCoordinates() {
    var experimentName = document.getElementById('experimentName').value;
    var sourceLat = document.getElementById('sourceLat').value;
    var sourceLong = document.getElementById('sourceLong').value;
    var targetLat = document.getElementById('targetLat').value;
    var targetLong = document.getElementById('targetLong').value;

    console.log("experiment Name: " + experimentName);
    console.log("Source Latitude: " + sourceLat);
    console.log("Source Longitude: " + sourceLong);
    console.log("Target Latitude: " + targetLat);
    console.log("Target Longitude: " + targetLong);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", currentUrl + "/submit_job");
    xhr.setRequestHeader("Content-Type", "application/json; charset=UTF-8")
    const body = JSON.stringify({
        exp_name: experimentName,
        s_lat: sourceLat,
        s_lon: sourceLong,
        t_lat: targetLat,
        t_lon: targetLong
    });
    xhr.onload = () => {
        if (xhr.readyState == 4) {
                alert(xhr.responseText);
        } else {
            console.log(`Status: ${xhr.status}`);
        }
    };
    xhr.send(body);
}

// getStatus();
// var intervalId = setInterval(
//     getStatus,
// 5000);

function submitAddrCoordinates() {
    // You can add any additional client-side validation here

    // Form will submit in a traditional way
    return true;
}