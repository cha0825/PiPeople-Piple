// upload 要比對的圖片
let image1Url = '';
let image2Url = '';

function uploadImage(imageBoxId, fileInputId) {
    const fileInput = document.getElementById(fileInputId);
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('box_id', imageBoxId);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // 上傳圖片＆顯示圖片
            document.getElementById(imageBoxId).innerHTML = '<img src="' + data.url + '" alt="Uploaded Image" style="width: 100%; height: 100%;">';
            if (imageBoxId === 'image-box1') {
                image1Url = data.url;
            } else if (imageBoxId === 'image-box2') {
                image2Url = data.url;
            }

        } else {
            alert('Image upload failed.');
        }
    })
    .catch(error => {
        console.error('Error uploading image:', error);
        alert('Image upload failed.');
    });
}

function handleSubmit() {
    if (image1Url && image2Url) {
        // 隐藏提交按钮
        document.getElementById('submit-button').style.display = 'none';
        // 显示 "running......" 信息
        document.getElementById('running-message').style.display = 'block';
        // 跑 runOverall.sh
        fetch('/run-script', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 整體比對結果寫入html
                console.log('SSIM HSV data Script executed successfully.');

                // Remove 'uploads/' from the image URLs
                const image1Path = image1Url.replace('uploads/', '');
                const image2Path = image2Url.replace('uploads/', '');
                
                // 傳送上傳的圖片跟ssim hsv結果之後跳轉到compare.html
                const randomDir = data.random_dir;
                const url = `/index/compare?image1=${encodeURIComponent(`/static/${randomDir}${image1Path}`)}&image2=${encodeURIComponent(`/static/${randomDir}${image2Path}`)}&ssim=${encodeURIComponent(data.ssim)}&hsv=${encodeURIComponent(data.hsv)}&cnn=${encodeURIComponent(data.cnn)}`;                
                window.location.href = url;
            } else {
                alert('Script execution failed.');
            }
        })
        .catch(error => {
            console.error('Error executing script:', error);
            alert('Script execution failed.');
        });
    } else {
        alert('Please upload both images before submitting.');
    }
}