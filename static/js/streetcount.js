// Display selected file names
document.getElementById('filePaths').addEventListener('change', function() {
    const files = this.files;
    const fileNames = [];
    for (let i = 0; i < files.length; i++) {
        fileNames.push(files[i].name);
    }
    document.getElementById('fileInput').innerHTML = `
        <label for="filePaths" class="form-label">Selected Files</label>
        <input type="text" class="form-control" id="filePaths" name="filePaths" value="${fileNames.join(', ')}" readonly>
    `;
});