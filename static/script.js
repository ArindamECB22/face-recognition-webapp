document.getElementById('uploadBtn').addEventListener('click', async () => {
    const input = document.getElementById('imageInput');
    if (!input.files[0]) {
        alert("Please select an image!");
        return;
    }

    const formData = new FormData();
    formData.append('image', input.files[0]);

    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });

    const data = await response.json();
    if (data.error) {
        alert(data.error);
        return;
    }

    document.getElementById('preview').src = '/' + data.image_url;
    document.getElementById('name').innerText = data.name;
});
