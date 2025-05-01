document.addEventListener('DOMContentLoaded', () => {
    // Select all purchase buttons and the product grid
    const purchaseBtns = document.querySelectorAll('.purchase-btn');
    const productGrid = document.querySelector('.product-grid'); // Select the grid
    const checkoutFormDiv = document.getElementById('checkout-form');
    const transactionForm = document.getElementById('transaction-form');
    const resultDiv = document.getElementById('result');
    const resultStatus = document.getElementById('result-status');
    const resultProbability = document.getElementById('result-probability');
    const formProductName = document.getElementById('form-product-name');
    const transactionAmountInput = document.getElementById('transaction-amount');
    const productCategorySelect = document.getElementById('product-category');
    const backToProductsBtn = document.getElementById('back-to-products-btn'); // Select the back button

    // Add click listeners to all purchase buttons
    purchaseBtns.forEach(button => {
        button.addEventListener('click', (event) => {
            const card = event.target.closest('.product-card');
            const productName = card.dataset.productName;
            const price = card.dataset.price;
            const category = card.dataset.category;

            // Populate form fields
            formProductName.textContent = productName;
            transactionAmountInput.value = price;
            // Set the product category dropdown
            // Check if the category exists as an option, otherwise select 'unknown' or default
            let categoryFound = false;
            for (let i = 0; i < productCategorySelect.options.length; i++) {
                if (productCategorySelect.options[i].value === category) {
                    productCategorySelect.value = category;
                    categoryFound = true;
                    break;
                }
            }
            if (!categoryFound) {
                // Handle case where category from card isn't in the dropdown
                // Option 1: Select a default like 'unknown' if it exists
                const unknownOption = Array.from(productCategorySelect.options).find(opt => opt.value.toLowerCase() === 'unknown');
                if (unknownOption) {
                    productCategorySelect.value = unknownOption.value;
                } else {
                    // Option 2: Add the category dynamically (more complex) or leave as default
                    // For simplicity, we'll leave it as the default first option if 'unknown' not found
                     productCategorySelect.selectedIndex = 0; // Select the first option
                }
                console.warn(`Product category '${category}' not found in dropdown. Selecting default.`);
            }


            // Show checkout form and hide the product grid
            checkoutFormDiv.classList.remove('hidden');
            productGrid.classList.add('hidden'); // Hide the grid
            resultDiv.classList.add('hidden'); // Hide previous result

            // Scroll to the form for better UX
            checkoutFormDiv.scrollIntoView({ behavior: 'smooth' });
        });
    });

    // Handle form submission
    transactionForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent default form submission

        // Clear previous results and show loading indicator (optional)
        resultStatus.textContent = 'Processing...';
        resultProbability.textContent = '';
        resultDiv.classList.remove('hidden');

        // Get form data
        const formData = new FormData(transactionForm);
        const data = {};
        formData.forEach((value, key) => {
            // Trim whitespace from text inputs
            if (typeof value === 'string') {
                data[key] = value.trim();
            } else {
                 data[key] = value;
            }
            // Handle potentially empty optional fields (like Transaction Hour/Date)
            // If empty, don't send them, let the backend handle defaults
            if (data[key] === "") {
                 delete data[key];
            }
        });

        // *** Automatically add the current hour ***
        data['Transaction Hour'] = new Date().getHours();

        // Convert specific fields expected as numbers by the backend/preprocessing
        const numericFields = ['Transaction Amount', 'Quantity', 'Customer Age', 'Account Age Days'];
        numericFields.forEach(field => {
            if (data.hasOwnProperty(field) && data[field] !== null && data[field] !== '') {
                const numValue = Number(data[field]);
                if (!isNaN(numValue)) {
                    data[field] = numValue;
                } else {
                    // Handle error or remove field if conversion fails
                    console.warn(`Could not convert field '${field}' to number: ${data[field]}`);
                    // Optionally delete data[field]; or let backend validation catch it
                }
            }
        });

        console.log('Sending data:', data); // Log data being sent

        try {
            // Send data to the backend /predict endpoint
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });

            const result = await response.json();
            console.log('Received result:', result); // Log result received

            if (response.ok) {
                // Display the result
                resultStatus.textContent = `Status: ${result.status}`;
                resultProbability.textContent = result.fraud_probability ? `Fraud Probability: ${result.fraud_probability}` : '';
                // Add styling based on result
                resultDiv.className = result.status.includes('Blocked') ? 'result-blocked' : 'result-approved';
            } else {
                // Display error message
                resultStatus.textContent = `Error: ${result.error || 'Prediction failed'}`;
                resultProbability.textContent = '';
                 resultDiv.className = 'result-error';
            }

        } catch (error) {
            console.error('Error submitting transaction:', error);
            resultStatus.textContent = 'Error: Could not connect to server.';
            resultProbability.textContent = '';
            resultDiv.className = 'result-error';
        } finally {
             // Optionally re-enable form or hide it
             // Instead of showing the original button, show the grid again
             // checkoutFormDiv.classList.add('hidden');
             // productGrid.classList.remove('hidden');
        }
    });

    // *** Add event listener for the back button ***
    backToProductsBtn.addEventListener('click', () => {
        checkoutFormDiv.classList.add('hidden'); // Hide the form
        productGrid.classList.remove('hidden'); // Show the product grid
        resultDiv.classList.add('hidden'); // Hide any previous result
        // Optional: Scroll back to the top or grid
        productGrid.scrollIntoView({ behavior: 'smooth' });
    });
});