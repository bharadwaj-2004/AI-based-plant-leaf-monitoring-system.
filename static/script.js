const fileInput =
    document.getElementById("fileInput");

const dropZone =
    document.getElementById("dropZone");

const previewContainer =
    document.getElementById("previewContainer");

const previewImage =
    document.getElementById("previewImage");

const scanButton =
    document.getElementById("scanButton");

const loading =
    document.getElementById("loading");

const resultCard =
    document.getElementById("resultCard");

const prediction =
    document.getElementById("prediction");

const confidence =
    document.getElementById("confidence");

const confidenceFill =
    document.getElementById("confidenceFill");

const predictionList =
    document.getElementById("predictionList");

const resetButton =
    document.getElementById("resetButton");


let selectedFile = null;


/* =========================================================
   FILE SELECT
========================================================= */

fileInput.addEventListener(
    "change",
    function () {

        if (
            this.files &&
            this.files.length > 0
        ) {

            setImage(
                this.files[0]
            );

        }

    }
);


/* =========================================================
   SET IMAGE
========================================================= */

function setImage(file) {

    if (!file.type.startsWith("image/")) {

        alert(
            "Please select an image file."
        );

        return;

    }


    selectedFile = file;


    const reader =
        new FileReader();


    reader.onload = function (event) {

        previewImage.src =
            event.target.result;

        previewContainer.classList.add(
            "show"
        );

        scanButton.disabled =
            false;

        resultCard.classList.remove(
            "show"
        );

    };


    reader.readAsDataURL(file);

}


/* =========================================================
   DRAG AND DROP
========================================================= */

dropZone.addEventListener(
    "dragover",
    function (event) {

        event.preventDefault();

        dropZone.classList.add(
            "dragging"
        );

    }
);


dropZone.addEventListener(
    "dragleave",
    function () {

        dropZone.classList.remove(
            "dragging"
        );

    }
);


dropZone.addEventListener(
    "drop",
    function (event) {

        event.preventDefault();

        dropZone.classList.remove(
            "dragging"
        );


        const files =
            event.dataTransfer.files;


        if (
            files &&
            files.length > 0
        ) {

            setImage(
                files[0]
            );

        }

    }
);


/* =========================================================
   CTRL + V IMAGE PASTE
========================================================= */

document.addEventListener(
    "paste",
    function (event) {

        const items =
            event.clipboardData.items;


        for (
            const item of items
        ) {

            if (
                item.type.startsWith(
                    "image/"
                )
            ) {

                const file =
                    item.getAsFile();


                if (file) {

                    setImage(
                        file
                    );

                }

                break;

            }

        }

    }
);


/* =========================================================
   SCAN
========================================================= */

scanButton.addEventListener(
    "click",
    async function () {

        if (!selectedFile) {

            return;

        }


        scanButton.disabled =
            true;

        loading.classList.add(
            "show"
        );


        const formData =
            new FormData();


        formData.append(
            "image",
            selectedFile
        );


        try {

            const response =
                await fetch(
                    "/predict",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.error ||
                    "Prediction failed."
                );

            }


            showResult(
                data
            );


        }

        catch (error) {

            alert(
                "Error: " +
                error.message
            );

        }

        finally {

            loading.classList.remove(
                "show"
            );

            scanButton.disabled =
                false;

        }

    }
);


/* =========================================================
   SHOW RESULT
========================================================= */

function showResult(data) {

    resultCard.classList.add(
        "show"
    );


    prediction.textContent =
        formatLabel(
            data.prediction
        );


    const confidencePercent =
        data.confidence * 100;


    confidence.textContent =
        confidencePercent.toFixed(2)
        + "%";


    setTimeout(
        function () {

            confidenceFill.style.width =
                confidencePercent
                + "%";

        },
        100
    );


    predictionList.innerHTML =
        "";


    data.predictions.forEach(
        function (item) {

            const percent =
                item.confidence * 100;


            const row =
                document.createElement(
                    "div"
                );


            row.className =
                "prediction-row";


            row.innerHTML = `

                <div class="prediction-name">
                    ${formatLabel(item.label)}
                </div>

                <div class="prediction-track">

                    <div
                        class="prediction-fill"
                        style="width: ${percent}%"
                    ></div>

                </div>

                <div class="prediction-value">
                    ${percent.toFixed(1)}%
                </div>

            `;


            predictionList.appendChild(
                row
            );

        }
    );


    resultCard.scrollIntoView({
        behavior: "smooth",
        block: "center"
    });

}


/* =========================================================
   FORMAT LABEL
========================================================= */

function formatLabel(label) {

    return label
        .replaceAll(
            "___",
            " — "
        )
        .replaceAll(
            "_",
            " "
        );

}


/* =========================================================
   RESET
========================================================= */

resetButton.addEventListener(
    "click",
    function () {

        selectedFile = null;

        fileInput.value = "";

        previewImage.src = "";

        previewContainer.classList.remove(
            "show"
        );

        resultCard.classList.remove(
            "show"
        );

        confidenceFill.style.width =
            "0%";

        predictionList.innerHTML =
            "";

        scanButton.disabled =
            true;

        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });

    }
);