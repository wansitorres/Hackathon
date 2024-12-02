# PropGuard

## Overview
PropGuard is a Python-based application designed to validate real estate listings by leveraging advanced AI agents. It ensures the authenticity and accuracy of property details, images, and reviews, providing a comprehensive analysis to potential buyers and real estate professionals.

## Functionalities
- **Property Validation**: Checks the validity of property details such as name, type, location, lot area, floor area, bedroom and bathroom counts, and price reasonability.
- **Image Validation**: Verifies the authenticity of property images, detecting any manipulations or duplicates.
- **Sentiment Analysis**: Analyzes property reviews to assess overall sentiment, highlighting key positive and negative points, as well as potential red flags.
- **Lister Verification**: Confirms the credentials of real estate agents by cross-referencing information across multiple trusted platforms.

## How It Works
1. **Initialization**: The agent is initialized with necessary API keys for Google Search and XAI services.
2. **Listing Validation**: Users provide a dictionary containing property details, which the agent processes to validate the listing.
3. **Task Creation**: The agent creates tasks for each validation aspect, assigning them to specialized AI agents (Property Validator, Image Validator, Sentiment Analyzer, and Lister Validator).
4. **Crew Execution**: A crew of agents executes the tasks in a specified order (sequential or hierarchical) to gather validation results.
5. **Result Formatting**: The results are formatted into a structured response, including a trust score, validation results, summary, and actionable recommendations.

## Purpose
The Real Estate Safety Agent aims to enhance the trustworthiness of real estate transactions by providing thorough validation of listings. It serves as a valuable tool for buyers, sellers, and real estate professionals to make informed decisions based on reliable data.
