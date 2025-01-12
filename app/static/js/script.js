class DrawingCanvas {
    constructor() {
        this.canvas = document.getElementById('drawingCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.isDrawing = false;
        this.setupCanvas();
        this.setupEventListeners();
    }

    setupCanvas() {
        // Make canvas size match its display size
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;

        // Set drawing style for MNIST compatibility
        this.ctx.strokeStyle = '#FFFFFF';
        this.ctx.lineWidth = Math.ceil(rect.width / 16); // Adjust line width based on canvas size
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
    }

    setupEventListeners() {
        // Mouse events
        this.canvas.addEventListener('mousedown', this.startDrawing.bind(this));
        this.canvas.addEventListener('mousemove', this.draw.bind(this));
        this.canvas.addEventListener('mouseup', this.stopDrawing.bind(this));
        this.canvas.addEventListener('mouseout', this.stopDrawing.bind(this));

        // Touch events for mobile
        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.startDrawing(this.getTouchPos(e));
        });
        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            this.draw(this.getTouchPos(e));
        });
        this.canvas.addEventListener('touchend', (e) => {
            e.preventDefault();
            this.stopDrawing();
        });

        // Window resize event
        window.addEventListener('resize', () => {
            const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
            const rect = this.canvas.getBoundingClientRect();
            this.canvas.width = rect.width;
            this.canvas.height = rect.height;
            this.ctx.putImageData(imageData, 0, 0);
            this.setupCanvas();
        });

        // Clear button
        const clearBtn = document.querySelector('.clear-btn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearCanvas());
        }

        // Predict button
        const predictBtn = document.querySelector('.predict-btn');
        if (predictBtn) {
            predictBtn.addEventListener('click', () => this.predict());
        }
    }

    startDrawing(e) {
        this.isDrawing = true;
        this.ctx.beginPath();
        const pos = this.getMousePos(e);
        this.ctx.moveTo(pos.x, pos.y);
    }

    draw(e) {
        if (!this.isDrawing) return;
        const pos = this.getMousePos(e);
        this.ctx.lineTo(pos.x, pos.y);
        this.ctx.stroke();
    }

    stopDrawing() {
        this.isDrawing = false;
    }

    getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }

    getTouchPos(e) {
        const rect = this.canvas.getBoundingClientRect();
        const touch = e.touches[0];
        return {
            clientX: touch.clientX - rect.left,
            clientY: touch.clientY - rect.top
        };
    }

    clearCanvas() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.updatePredictionDisplay('-', 0);
        this.updateDistributionBars(new Array(10).fill(0));
    }

    async predict() {
        // Show processing overlay
        const overlay = document.querySelector('.processing-overlay');
        overlay.style.display = 'flex';

        try {
            // Get the image data
            const imageData = this.canvas.toDataURL('image/png');

            // Send to server
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });

            const result = await response.json();

            if (result.success) {
                // Update UI with prediction results
                this.updatePredictionDisplay(result.prediction, result.confidence);
                this.updateDistributionBars(result.probabilities);
                this.updateMetrics(result.confidence);
            } else {
                console.error('Prediction failed:', result.error);
                alert('Prediction failed. Please try again.');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred. Please try again.');
        } finally {
            // Hide processing overlay
            overlay.style.display = 'none';
        }
    }

    updatePredictionDisplay(digit, confidence) {
        const digitDisplay = document.querySelector('.digit');
        const confidenceFill = document.querySelector('.confidence-fill');
        const confidenceText = document.querySelector('.confidence-text');

        digitDisplay.textContent = digit;
        confidenceFill.style.width = `${confidence}%`;
        confidenceText.textContent = `Confidence: ${confidence.toFixed(1)}%`;
    }

    updateDistributionBars(distribution) {
        const barsContainer = document.querySelector('.distribution-bars');
        barsContainer.innerHTML = '';

        const digitNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

        distribution.forEach((prob, i) => {
            const bar = document.createElement('div');
            bar.className = 'distribution-bar';
            
            const fill = document.createElement('div');
            fill.className = 'distribution-fill';
            fill.style.height = `${prob * 100}%`;
            
            const label = document.createElement('div');
            label.className = 'distribution-label';
            label.textContent = digitNames[i];

            const value = document.createElement('div');
            value.className = 'distribution-value';
            value.textContent = `${(prob * 100).toFixed(1)}%`;

            bar.appendChild(fill);
            bar.appendChild(label);
            bar.appendChild(value);
            barsContainer.appendChild(bar);
        });
    }

    updateMetrics(confidence) {
        const confidenceMetric = document.querySelectorAll('.metric-value')[1];
        if (confidenceMetric) {
            confidenceMetric.textContent = `${confidence.toFixed(1)}%`;
        }
    }
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new DrawingCanvas();
});
