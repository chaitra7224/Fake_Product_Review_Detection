/* =========================================================
   FAKE REVIEW DETECTION
   COMMON JAVASCRIPT
========================================================= */


/* =========================================================
   PAGE LOAD
========================================================= */

document.addEventListener("DOMContentLoaded", function () {

    initializeFileInput();

    initializeAnalyticsCharts();

});


/* =========================================================
   DATASET FILE INPUT
========================================================= */

function initializeFileInput() {

    const fileInput = document.getElementById("datasetFile");

    const fileName = document.getElementById("fileName");


    /* Stop if the page does not contain file input */

    if (!fileInput || !fileName) {
        return;
    }


    /* Display selected file name */

    fileInput.addEventListener("change", function () {

        if (fileInput.files.length > 0) {

            const selectedFile = fileInput.files[0];

            fileName.textContent =
                "Selected file: " + selectedFile.name;

        } else {

            fileName.textContent =
                "No file selected";

        }

    });

}


/* =========================================================
   ANALYTICS CHARTS
========================================================= */

function initializeAnalyticsCharts() {

    /* Check whether Chart.js is available */

    if (typeof Chart === "undefined") {
        return;
    }


    /* Check whether analytics data exists */

    if (typeof window.analyticsData === "undefined") {
        return;
    }


    /* =====================================================
       MODEL ACCURACY CHART
    ===================================================== */

    const accuracyCanvas =
        document.getElementById("accuracyChart");


    if (accuracyCanvas) {

        new Chart(
            accuracyCanvas,
            {

                type: "bar",

                data: {

                    labels: [
                        "Decision Tree",
                        "AdaBoost"
                    ],

                    datasets: [
                        {

                            label: "Accuracy (%)",

                            data: [
                                window.analyticsData.decisionTreeAccuracy,
                                window.analyticsData.adaBoostAccuracy
                            ],

                            borderWidth: 1

                        }
                    ]

                },

                options: {

                    responsive: true,

                    maintainAspectRatio: false,

                    scales: {

                        y: {

                            beginAtZero: true,

                            max: 100,

                            title: {

                                display: true,

                                text: "Accuracy (%)",

                                color: "#cccccc"

                            },

                            ticks: {

                                color: "#cccccc"

                            },

                            grid: {

                                color: "#333333"

                            }

                        },

                        x: {

                            ticks: {

                                color: "#cccccc"

                            },

                            grid: {

                                color: "#333333"

                            }

                        }

                    },

                    plugins: {

                        legend: {

                            display: false

                        }

                    }

                }

            }
        );

    }


    /* =====================================================
       DATASET CLASS DISTRIBUTION
    ===================================================== */

    const classCanvas =
        document.getElementById("classChart");


    if (classCanvas) {

        new Chart(
            classCanvas,
            {

                type: "pie",

                data: {

                    labels: [
                        "Computer Generated (CG)",
                        "Original (OR)"
                    ],

                    datasets: [
                        {

                            data: [
                                window.analyticsData.fakeReviews,
                                window.analyticsData.originalReviews
                            ],

                            borderWidth: 1

                        }
                    ]

                },

                options: {

                    responsive: true,

                    maintainAspectRatio: false,

                    plugins: {

                        legend: {

                            position: "bottom",

                            labels: {

                                color: "#cccccc"

                            }

                        }

                    }

                }

            }
        );

    }


    /* =====================================================
       RATING VS AUTHENTICITY
    ===================================================== */

    const ratingCanvas =
        document.getElementById("ratingChart");


    if (ratingCanvas) {

        const ratingData =
            window.analyticsData.ratingData || [];


        const ratingLabels =
            ratingData.map(function (item) {

                return item.rating;

            });


        const cgData =
            ratingData.map(function (item) {

                return item.CG;

            });


        const orData =
            ratingData.map(function (item) {

                return item.OR;

            });


        new Chart(
            ratingCanvas,
            {

                type: "bar",

                data: {

                    labels: ratingLabels,

                    datasets: [

                        {

                            label: "Computer Generated (CG)",

                            data: cgData,

                            borderWidth: 1

                        },

                        {

                            label: "Original (OR)",

                            data: orData,

                            borderWidth: 1

                        }

                    ]

                },

                options: {

                    responsive: true,

                    maintainAspectRatio: false,

                    scales: {

                        x: {

                            title: {

                                display: true,

                                text: "Rating",

                                color: "#cccccc"

                            },

                            ticks: {

                                color: "#cccccc"

                            },

                            grid: {

                                color: "#333333"

                            }

                        },

                        y: {

                            beginAtZero: true,

                            title: {

                                display: true,

                                text: "Number of Reviews",

                                color: "#cccccc"

                            },

                            ticks: {

                                color: "#cccccc"

                            },

                            grid: {

                                color: "#333333"

                            }

                        }

                    },

                    plugins: {

                        legend: {

                            labels: {

                                color: "#cccccc"

                            }

                        }

                    }

                }

            }
        );

    }

}