let session_id = '';
    let triplines = [];
    let currentTripline = {};
    let videoUploaded = false;

    document.getElementById('videoFile').addEventListener('change', function () {
        const videoFile = this.files[0];
        if (videoFile) {
            const formData = new FormData();
            formData.append('videoFile', videoFile);


            fetch('/initialize', {
                method: 'POST',
                body: formData,
            })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        session_id = data.session_id;
                        loadCanvasImage(data.frame_url);
                        document.getElementById('triplineSection').style.display = 'block';
                        videoUploaded = true;
                    } else {
                        alert('Error uploading video: ' + data.message);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('An error occurred while uploading the video.');
                });
        }
    });

    function loadCanvasImage(imageUrl) {
        const canvas = document.getElementById('drawCanvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();
        img.src = imageUrl;
        img.onload = function () {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);

            initCanvasEvents(img);
        };
    }

    function initCanvasEvents(img) {
        const canvas = document.getElementById('drawCanvas');
        const ctx = canvas.getContext('2d');
        let isDrawing = false;
        currentTripline = {};

        function getScaledCoordinates(e) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            return {
                x: (e.clientX - rect.left) * scaleX,
                y: (e.clientY - rect.top) * scaleY
            };
        }

        canvas.addEventListener('mousedown', (e) => {
            currentTripline.start = getScaledCoordinates(e);
            isDrawing = true;
        });

        canvas.addEventListener('mousemove', (e) => {
            if (!isDrawing) return;
            currentTripline.end = getScaledCoordinates(e);

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);

            triplines.forEach((line, index) => {
                drawLine(ctx, line);
                drawIndex(ctx, index, line);
            });

            drawLine(ctx, currentTripline);
        });

        canvas.addEventListener('mouseup', () => {
            isDrawing = false;
            if (currentTripline.start && currentTripline.end) {
                triplines.push({ ...currentTripline });
                currentTripline = {};
                updateTriplineCounter();

                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0);
                triplines.forEach((line, index) => {
                    drawLine(ctx, line);
                    drawIndex(ctx, index, line);
                });
            }
        });

        window.addEventListener('resize', () => {
            // Redraw frame and all triplines on window resize
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);
            triplines.forEach(line, index => {
                drawLine(ctx, line);
                drawIndex(ctx, index, line);
            });
        });

        document.getElementById('resetTriplinesBtn').addEventListener('click', () => {
            triplines = [];
            currentTripline = {};

            // Update the tripline counter
            updateTriplineCounter();

            // Clear canvas  redraw  original image
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);

            document.getElementById('directionsCard').style.display = 'none';
            document.getElementById('resetTriplinesBtn').style.display = 'none';
        });
    }

    function drawLine(ctx, line) {
        ctx.beginPath();
        ctx.moveTo(line.start.x, line.start.y);
        ctx.lineTo(line.end.x, line.end.y);
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    function drawIndex(ctx, index, line) {
        ctx.fillStyle = 'red';
        ctx.font = '24px sans-serif';
        const m_x = (line.start.x + line.end.x) / 2;
        const m_y = (line.start.y + line.end.y) / 2;
        const dx = line.end.x - line.start.x;
        const dy = line.end.y - line.start.y;
        const length = Math.sqrt(dx * dx + dy * dy);
        const orthogonal_x = dy / length;
        const orthogonal_y = -dx / length;
        const offset = 20;
        const pt_x = m_x + orthogonal_x * offset;
        const pt_y = m_y + orthogonal_y * offset;
        ctx.fillText(index + 1, pt_x, pt_y);
    }

    function updateTriplineCounter() {
        document.getElementById('triplineCount').innerText = triplines.length;
        if (triplines.length > 0) {
            document.getElementById('directionsCard').style.display = 'block';
            adjustDirectionsInputs(); 
            document.getElementById('resetTriplinesBtn').style.display = 'inline-block';
        } else {
            document.getElementById('directionsCard').style.display = 'none';
            document.getElementById('processingSection').style.display = 'none';
        }
    }

    function adjustDirectionsInputs() {
        const directionsForm = document.getElementById('directionsForm');
        // Clear existing direction inputs
        directionsForm.innerHTML = '';

        if (triplines.length === 1) {
            // If one tripline, ask for 2 directions
            addDirectionInput(directionsForm, 'direction1', 'Direction 1');
            addDirectionInput(directionsForm, 'direction2', 'Direction 2');
        } else if (triplines.length >= 2) {
            // If multiple triplines, ask for 1 dir/tripline
            triplines.forEach((tripline, index) => {
                addDirectionInput(directionsForm, `direction${index + 1}`, `Direction for Tripline ${index + 1}`);
            });
        }
    }



    function addDirectionInput(form, id, label) {
        const div = document.createElement('div');
        div.className = 'mb-3';

        const labelElem = document.createElement('label');
        labelElem.htmlFor = id;
        labelElem.className = 'form-label';
        labelElem.innerText = label;

        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'form-control';
        input.id = id;
        input.name = id;
        input.required = true;

        div.appendChild(labelElem);
        div.appendChild(input);
        form.appendChild(div);
    }

    function copyToClipboard(text) {
        navigator.clipboard.writeText(text).then(function () {
            const feedback = document.getElementById('copyFeedback');
            feedback.style.display = 'inline';
            setTimeout(() => {
                feedback.style.display = 'none';
            }, 1000);
        }, function (err) {
            console.error('Could not copy text: ', err);
        });
    }

    document.addEventListener('click', function (event) {
        if (event.target && event.target.id === 'sessionId') {
            const sessionId = event.target.textContent;
            copyToClipboard(sessionId);
        }
    });

    document.getElementById('saveAndSubmitBtn').addEventListener('click', () => {
        if (triplines.length === 0) {
            alert('Please draw at least one tripline before saving and submitting.');
            return;
        }

        let directions = {};
        if (triplines.length === 1) {
            const directionValue1 = document.getElementById(`direction1`).value;
            const directionValue2 = document.getElementById(`direction2`).value;
            directions = {
                1: directionValue1,
                2: directionValue2
            };
        } else if (triplines.length >= 2) {
            triplines.forEach((tripline, index) => {
                const directionValue = document.getElementById(`direction${index + 1}`).value;
                directions[index + 1] = directionValue;
            });
        }

        // Include directions in the form data
        const form = document.getElementById('inputForm');
        const formData = new FormData(form);
        formData.append('directions', JSON.stringify(directions));
        formData.append('triplines', JSON.stringify(triplines));

        // Show progress bars and reset them
        const progressBarYOLO = document.getElementById('progressBarYOLO');
        const progressBarCounting = document.getElementById('progressBarCounting');
        const progressBarExcel = document.getElementById('progressBarExcel');
        const progressBarAnnotation = document.getElementById('progressBarAnnotation');

        // Reset progress bars
        progressBarYOLO.style.width = '0%';
        progressBarYOLO.innerText = 'YOLO: 0%';

        progressBarCounting.style.width = '0%';
        progressBarCounting.innerText = 'Counting: 0%';

        progressBarExcel.style.width = '0%';
        progressBarExcel.innerText = 'Report: 0%';

        progressBarAnnotation.style.width = '0%';
        progressBarAnnotation.innerText = 'Annotation: 0%';

        // Display progress bars
        document.querySelectorAll('.progress').forEach(progress => {
            progress.style.display = 'block';
        });

        document.getElementById('result').innerText = '';
        document.getElementById('downloadLinks').innerHTML = '';

        // get the session_id
        fetch(`/pre_process/${session_id}`, {
            method: 'POST',
            body: formData,
        })
            .then(response => response.json())
            .then(data => {
                if (data.status === "success") {
                    fetch(`/start_processing/${session_id}`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ 'triplines': triplines, 'directions': directions })
                    })
                        .then(response => response.json())
                        .then(data => {
                            if (data.status === "Processing started") {
                                document.getElementById('result').innerHTML = `
                                Please wait, processing...<br>
                                Session ID : <span id="sessionId" class="text-primary" style="cursor: pointer; text-decoration: underline;">
                                    ${session_id}
                                </span>
                                <span id="copyFeedback" class="text-success" style="display: none; margin-left: 10px;">
                                    Copied!
                                </span>
                            `;

                                // Start polling for progress updates
                                const progressInterval = setInterval(() => {
                                    fetch(`/progress?session_id=${session_id}`)
                                        .then(response => response.json())
                                        .then(progressData => {
                                            if (progressData.YOLO >= 0 && progressData.YOLO <= 100) {
                                                progressBarYOLO.style.width = progressData.YOLO + '%';
                                                progressBarYOLO.innerText = 'YOLO: ' + progressData.YOLO + '%';
                                            }
                                            if (progressData.Counting >= 0 && progressData.Counting <= 100) {
                                                progressBarCounting.style.width = progressData.Counting + '%';
                                                progressBarCounting.innerText = 'Counting: ' + progressData.Counting + '%';
                                            }
                                            if (progressData.Excel >= 0 && progressData.Excel <= 100) {
                                                progressBarExcel.style.width = progressData.Excel + '%';
                                                progressBarExcel.innerText = 'Report: ' + progressData.Excel + '%';
                                            }
                                            if (progressData.Annotation >= 0 && progressData.Annotation <= 100 && document.getElementById('exportVideo').checked) {
                                                progressBarAnnotation.style.width = progressData.Annotation + '%';
                                                progressBarAnnotation.innerText = 'Annotation: ' + progressData.Annotation + '%';
                                            }
                                            // Check if processing is complete
                                            const isComplete = (progressData.YOLO === 100 &&
                                                progressData.Counting === 100 &&
                                                progressData.Excel === 100 &&
                                                (progressData.Annotation === 100 || !document.getElementById('exportVideo').checked));

                                            if (isComplete) {
                                                progressBarYOLO.className = "progress-bar progress-bar-striped progress-bar-good"
                                                progressBarCounting.className = "progress-bar progress-bar-striped progress-bar-good"
                                                progressBarExcel.className = "progress-bar progress-bar-striped progress-bar-good"
                                                progressBarAnnotation.className = "progress-bar progress-bar-striped progress-bar-good"
                                                clearInterval(progressInterval);
                                                document.getElementById('result').innerHTML = `
                                Processing complete. <br>
                                Session ID : <span id="sessionId" class="text-primary" style="cursor: pointer; text-decoration: underline;">
                                    ${session_id}
                                </span>
                                <span id="copyFeedback" class="text-success" style="display: none; margin-left: 10px;">
                                    Copied!
                                </span>
                            `;
                                                // Generate download links
                                                const downloadLinks = [];
                                                downloadLinks.push(`<a href="/download/${session_id}/${data.paths.report_path}" class="btn btn-success m-2">Download Report</a>`);

                                                // Add video download link if video export was enabled
                                                if (document.getElementById('exportVideo').checked) {
                                                    downloadLinks.push(`<a href="/download/${session_id}/${data.paths.annotated_video_path}" class="btn btn-success m-2">Download Annotated Video</a>`);
                                                }
                                                else {
                                                    downloadLinks.push(' <a href="#" class="btn btn-secondary m-2 disabled">No video output</a>');
                                                }
                                                document.getElementById('downloadLinks').innerHTML = downloadLinks.join('');
                                            }
                                            // Check for errors
                                            else if (progressData.YOLO === -1 ||
                                                progressData.Counting === -1 ||
                                                progressData.Excel === -1 ||
                                                progressData.Annotation === -1) {
                                                progressBarYOLO.className = "progress-bar progress-bar-striped progress-bar-bad"
                                                progressBarCounting.className = "progress-bar progress-bar-striped progress-bar-bad"
                                                progressBarExcel.className = "progress-bar progress-bar-striped progress-bar-bad"
                                                progressBarAnnotation.className = "progress-bar progress-bar-striped progress-bar-bad"
                                                clearInterval(progressInterval);
                                                document.getElementById('result').innerText = 'An error occurred during processing.';
                                                // Try to fetch specific error message
                                                fetch(`/results?session_id=${session_id}`)
                                                    .then(response => response.json())
                                                    .then(resultData => {
                                                        if (resultData.error) {
                                                            document.getElementById('result').innerText = `Error: ${resultData.error}`;
                                                        }
                                                    });
                                            }
                                        })
                                        .catch(error => {
                                            console.error('Error:', error);
                                            clearInterval(progressInterval);
                                            document.getElementById('result').innerText = 'An unexpected error occurred.';
                                        });
                                }, 500); // Poll every half second
                            } else {
                                document.getElementById('result').innerText = data.error || 'An unexpected error occurred';
                            }
                        });
                } else {
                    document.getElementById('result').innerText = data.error || 'An unexpected error occurred';
                }
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('result').innerText = error || 'An unexpected error occurred';
            });

        alert('Triplines and directions saved, processing started.');
    });
