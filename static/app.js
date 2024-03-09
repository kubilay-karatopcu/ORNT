document.addEventListener('DOMContentLoaded', function() {
    fetch('/get-plots')
        .then(response => response.json())
        .then(plots => {
            // Iterate over each plot configuration received from the server
            plots.forEach((plotData, index) => {
                // The index is 0-based, but your div IDs are 1-based, hence `index + 1`
                const plotId = `plot-${index + 1}`;
                let explanationId = `explanation${index + 1}`;

                // Select the div where the plot should be rendered
                const plotDiv = document.getElementById(plotId);

                // Check if plotData.plot is already an object or a string
                const plotConfig = typeof plotData.plot === 'string' ? JSON.parse(plotData.plot) : plotData.plot;

                // Check if the div exists before attempting to render the plot
                if (plotDiv) {
                    Plotly.newPlot(plotDiv, plotConfig);
                }

                const explanationDiv = document.getElementById(explanationId);
                if (explanationDiv) {
                    explanationDiv.textContent = plotData.explanation;
                }
            });
        });
});