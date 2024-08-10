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
            document.getElementById(imageBoxId).innerHTML = '<img src="' + data.url + '" alt="Uploaded Image" style="width: 100%; height: 100%;">';

            // 檢查兩個圖片是否都已經上傳
            const imageBox1 = document.getElementById('image-box1');
            const imageBox2 = document.getElementById('image-box2');
            if (imageBox1.innerHTML.includes('<img') && imageBox2.innerHTML.includes('<img')) {
                // 发送请求执行 .sh 文件
                fetch('/run-script', {
                    method: 'POST'
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('ssim-result').innerText = data.ssim;
                        document.getElementById('hsv-result').innerText = data.hsv;
                        console.log('SSIM HSV data Script executed successfully.');
                        
                        // 寫局部比對的特徵的選項 這裡是要科出那個html
                        fetch('/get_features', {
                            method: 'POST'
                        })
                        .then(response => response.json())
                        .then(data => {
                            const container = document.getElementById('id-result-container');

                            // Create the first row description
                            const description1 = document.createElement('div');
                            description1.className = 'feature-images-description';
                            description1.textContent = 'image1';

                            // Create the first row container
                            const container1 = document.createElement('div');
                            container1.className = 'result-container';
                            container1.appendChild(description1);

                            // Check if features1 is empty
                            if (data.features1.length === 0) {
                                const noFeaturesMessage1 = document.createElement('div');
                                noFeaturesMessage1.className = 'no-features-message';
                                noFeaturesMessage1.textContent = 'This image has no features detected.';
                                container1.appendChild(noFeaturesMessage1);
                            } else {

                            // Create the first row
                            try {
                                const row1 = document.createElement('div');
                                row1.className = 'feature-selection-row1';
                                data.features1.forEach((feature, index) => {
                                    try {
                                        const item = document.createElement('div');
                                        item.className = 'feature-selection-item';

                                        const checkbox = document.createElement('input');
                                        checkbox.type = 'checkbox';
                                        checkbox.id = `feature1_${index}`;
                                        checkbox.name = `feature1_${feature}`;

                                        const label = document.createElement('label');
                                        label.htmlFor = `feature1_${index}`;

                                        const img = document.createElement('img');
                                        img.src = `static/partpic/detection_${index}_${feature}.jpg`;
                                        img.alt = feature;

                                        // Print the image source and feature for debugging
                                        console.log(`Row 1 - Image Source: ${img.src}, Feature: ${feature}`);

                                        label.appendChild(img);
                                        item.appendChild(checkbox);
                                        item.appendChild(label);
                                        row1.appendChild(item);
                                    } catch (error) {
                                        console.error(`Error processing feature1 index ${index}:`, error);
                                    }
                                });
                                container1.appendChild(row1);
                            } catch (error) {
                                console.error('Error initializing feature selection rows:', error);
                            }
                        }
                            container.appendChild(container1);
                            
                            /*
                            // Create the divider
                            const divider = document.createElement('div');
                            divider.className = 'feature-divider';
                            container.appendChild(divider);
                            */

                            // Create the second row description
                            const description2 = document.createElement('div');
                            description2.className = 'feature-images-description';
                            description2.textContent = 'image2';

                            // Create the second row container
                            const container2 = document.createElement('div');
                            container2.className = 'result-container';
                            container2.appendChild(description2);
                        
                            // Check if features2 is empty
                            if (data.features2.length === 0) {
                                const noFeaturesMessage2 = document.createElement('div');
                                noFeaturesMessage2.className = 'no-features-message';
                                noFeaturesMessage2.textContent = 'This image has no features detected.';
                                container2.appendChild(noFeaturesMessage2);
                            } else {

                                // Create the second row
                                const row2 = document.createElement('div');
                                row2.className = 'feature-selection-row2';
                                data.features2.forEach((feature, index) => {
                                    try {
                                        const item = document.createElement('div');
                                        item.className = 'feature-selection-item';

                                        const checkbox = document.createElement('input');
                                        checkbox.type = 'checkbox';
                                        checkbox.id = `feature2_${index}`;
                                        checkbox.name = `feature2_${feature}`;

                                        const label = document.createElement('label');
                                        label.htmlFor = `feature2_${index}`;

                                        const img = document.createElement('img');
                                        img.src = `static/partpic/detection_${index}_${feature}.jpg`;
                                        img.alt = feature;

                                        // Print the image source and feature for debugging
                                        console.log(`Row 2 - Image Source: ${img.src}, Feature: ${feature}`);

                                        label.appendChild(img);
                                        item.appendChild(checkbox);
                                        item.appendChild(label);
                                        row2.appendChild(item);
                                    } catch (error) {
                                        console.error(`Error processing feature2 index ${index}:`, error);
                                    }
                                });
                                container2.appendChild(row2);
                            }
                            container.appendChild(container2);

                            // Check if features1 and features2 are not empty before showing the submit button
                            if (data.features1.length > 0 && data.features2.length > 0) {
                                document.getElementById('submit-button-container').style.display = 'block';
                            }

                            // 全部都跑完之後 做顯示
                            document.getElementById('post-upload-content').style.display = 'block';
                            
                            // 因為局部特徵只能單選 這裡是做觸發器 讓他偵測
                            handleRowCheckboxes('.feature-selection-row1');
                            handleRowCheckboxes('.feature-selection-row2');
                        })
                    } else {
                        alert('Script execution failed.');
                    }
                })
                .catch(error => {
                    console.error('Error executing script:', error);
                    alert('Script execution failed.');
                });
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
/*當另一個checkbox 被點選時，已點選的那個checkbox 會自己取消點選*/
function handleRowCheckboxes(rowSelector) {
    const rows = document.querySelectorAll(rowSelector);
    rows.forEach(row => {
        console.log(row.innerHTML);
        
        const checkboxes = row.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(checkbox => {
            checkbox.addEventListener('change', function(e) {
                console.log('eventlistener is called');
                if (e.target.checked) {
                    checkboxes.forEach(otherCheckbox => {
                        if (otherCheckbox !== e.target) {
                            otherCheckbox.checked = false;
                        }
                    });
                }
            });
        });
    });
}