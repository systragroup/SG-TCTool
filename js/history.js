document.addEventListener('DOMContentLoaded', function () {
    {% if selected_type == 'Counting' and selected_session_id and first_frame_path and triplines %}
    const frameUrl = "{{ first_frame_path }}";
    const triplines = {{ triplines | tojson }};
    console.log(triplines)
    loadCanvasImage(frameUrl, triplines);
    {% endif %}
    });

function loadCanvasImage(imageUrl, triplines) {
    const canvas = document.getElementById('annotatedCanvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.src = imageUrl;
    img.onload = function () {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);

        // Now draw triplines
        triplines.forEach((line, index) => {
            drawLine(ctx, line);
            drawIndex(ctx, index, line);
        });
    };
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