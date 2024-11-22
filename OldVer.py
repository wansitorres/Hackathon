from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import Dict, List
import json
import requests
from PIL import Image
from io import BytesIO
import imagehash

class RealEstateSafetyAgent:
    def __init__(self, openai_api_key: str, google_api_key: str, google_cse_id: str):
        # Initialize the LLM
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4",
            openai_api_key=openai_api_key
        )
        
        # Initialize tools
        self.search = GoogleSearchAPIWrapper(
            google_api_key=google_api_key,
            google_cse_id=google_cse_id
        )
        
        # Create tools list
        self.tools = [
            Tool(
                name="Search",
                func=self.search.run,
                description="Useful for searching information about properties and listings online"
            ),
            Tool(
                name="ValidateImages",
                func=self.validate_images,
                description="Validates property images for authenticity"
            ),
            Tool(
                name="AnalyzeSentiment",
                func=self.analyze_sentiment,
                description="Analyzes sentiment from property reviews"
            ),
            Tool(
                name="CrossPlatformCheck",
                func=self.check_cross_platform,
                description="Checks if property exists on other platforms"
            )
        ]
        
        # Create the agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self._create_prompt()
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True
        )

    def validate_listing(self, listing_data: Dict) -> Dict:
        """Main method to validate a real estate listing"""
        result = self.agent_executor.invoke({
            "input": f"Please analyze this property listing: {json.dumps(listing_data)}"
        })
        
        return self._format_validation_result(result)

    def validate_images(self, image_urls: List[str]) -> Dict:
        """Validates images for authenticity and duplicates"""
        results = {
            'duplicates': [],
            'suspicious': [],
            'valid': [],
            'metadata_issues': []
        }
        
        image_hashes = []
        
        for url in image_urls:
            try:
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
                img_hash = str(imagehash.average_hash(img))
                
                # Check for duplicates
                if any(self._is_similar(img_hash, h) for h in image_hashes):
                    results['duplicates'].append(url)
                else:
                    results['valid'].append(url)
                    image_hashes.append(img_hash)
                    
            except Exception as e:
                results['suspicious'].append({'url': url, 'error': str(e)})
                
        return results

    def analyze_sentiment(self, reviews: List[str]) -> Dict:
        """Analyzes sentiment from property reviews"""
        analysis_prompt = """
        Analyze the sentiment of these reviews and provide:
        1. Overall sentiment score (0-1)
        2. Key positive points
        3. Key negative points
        4. Potential red flags
        
        Reviews: {reviews}
        """
        
        result = self.llm.invoke(analysis_prompt.format(reviews=reviews))
        return json.loads(result.content)

    def check_cross_platform(self, listing_data: Dict) -> Dict:
        """Checks for the same property listing across different platforms"""
        search_query = f"{listing_data['address']} {listing_data['price']} real estate listing"
        search_results = self.search.run(search_query)
        
        return {
            'cross_platform_listings': search_results,
            'platforms_found': self._extract_platforms(search_results)
        }

    def _is_similar(self, hash1: str, hash2: str, threshold: int = 8) -> bool:
        """Checks if two image hashes are similar"""
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2)) < threshold

    def _extract_platforms(self, search_results: str) -> List[str]:
        """Extracts platform names from search results"""
        common_platforms = ['zillow', 'trulia', 'realtor.com', 'redfin', 'facebook marketplace']
        found_platforms = []
        
        for platform in common_platforms:
            if platform.lower() in search_results.lower():
                found_platforms.append(platform)
                
        return found_platforms

    def _create_prompt(self) -> PromptTemplate:
        """Creates the agent prompt template"""
        return PromptTemplate.from_template(
            """You are a real estate safety expert AI agent. Your goal is to validate property 
            listings and identify potential fraud or suspicious activity.
            
            You have access to the following tools:
            {tools}
            
            Available tool names: {tool_names}
            
            Analyze the given property listing using the available tools and provide a detailed 
            safety report including:
            1. Image validation results
            2. Sentiment analysis from reviews
            3. Cross-platform presence
            4. Overall trust score
            5. Potential red flags
            
            Property listing: {input}
            
            Think through this step-by-step:
            1. First, validate the images
            2. Then, analyze any available reviews
            3. Check for cross-platform listings
            4. Synthesize all information for a final assessment
            
            {agent_scratchpad}
            """
        )

    def _format_validation_result(self, agent_result: Dict) -> Dict:
        """Formats the agent's output into a structured response"""
        return {
            'trust_score': self._calculate_trust_score(agent_result),
            'validation_results': agent_result,
            'summary': self._generate_summary(agent_result),
            'recommendations': self._generate_recommendations(agent_result)
        }

    def _calculate_trust_score(self, results: Dict) -> float:
        """Calculates an overall trust score based on all validation results"""
        # Implementation of trust score calculation
        # This would be based on various factors from the validation results
        pass

    def _generate_summary(self, results: Dict) -> str:
        """Generates a human-readable summary of the validation results"""
        # Implementation of summary generation
        pass

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generates actionable recommendations based on validation results"""
        # Implementation of recommendations generation
        pass

# Initialize the agent
agent = RealEstateSafetyAgent(openai_api_key="", google_api_key="", google_cse_id="")

listing_data = {
    "address": "123 Main St, City, State",
    "price": 500000,
    "description": "Beautiful 3-bedroom house...",
    "images": ["https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.rocketmortgage.com%2Flearn%2Freal-property&psig=AOvVaw3P5J9i0k4hWIYzhf5qc4Dh&ust=1732362923708000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCJCp7ffw74kDFQAAAAAdAAAAABAE"],
    "reviews": ["Great property!", "The location is perfect"],
}

# Validate the listing
results = agent.validate_listing(listing_data)
print(results)
