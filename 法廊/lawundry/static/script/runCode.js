window.onload = function() {
    // run runPartial.sh
    fetch('/get_features', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        // display 局部比對結果
        const container = document.getElementById('id-result-container');

        // create the first row description
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
                        img.src = `/static/partpic/detection_img1_${index}_${feature}.jpg`;
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
                    img.src = `/static/partpic/detection_img2_${index}_${feature}.jpg`;
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

        // 跑完之後顯示結果
        document.getElementById('post-featrue-content').style.display = 'block';
        
        // 看會不會進入局部比對階段
        if (data.features1.length > 0 && data.features2.length > 0) {
            document.getElementById('submit-button-container').style.display = 'block';
        }
        // 因為局部特徵只能單選 這裡是做觸發器 讓他偵測
        handleRowCheckboxes('.feature-selection-row1');
        handleRowCheckboxes('.feature-selection-row2');

    });
}


/*當另一個checkbox 被點選時，已點選的那個checkbox 會自己取消點選*/
function handleRowCheckboxes(rowSelector) {
    console.log(`${rowSelector} is called`);
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