from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from typing import List, Dict
import requests
from PIL import Image
import imagehash
from io import BytesIO
import json
from urllib.parse import quote_plus

class RealEstateSafetyAgent:
    def __init__(self, openai_api_key: str):
        # Initialize the LLM
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4",
            openai_api_key=openai_api_key
        )
        
        # Initialize tools
        self.search = DuckDuckGoSearchRun()
        
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
            ),
            Tool(
                name="AnalyzeRating",
                func=self.analyze_rating,
                description="Analyzes the average review rating (1-5 scale)"
            ),
            Tool(
                name="VerifyUser",
                func=self.verify_user,
                description="Verifies user credibility across platforms"
            ),
            Tool(
                name="GetUserListings",
                func=self.get_user_listings,
                description="Retrieves user's listings from other platforms"
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

    def verify_user(self, username: str) -> Dict:
        """Verifies user credibility across different platforms"""
        search_query = f"{username} real estate agent listings reviews"
        search_results = self.search.run(search_query)
        
        results = {
            'username': username,
            'platform_presence': {},
            'total_listings_found': 0,
            'review_summary': [],
            'risk_level': 'unknown'
        }
        
        # Common real estate platforms to check
        platforms = ['zillow', 'trulia', 'realtor.com', 'redfin', 'facebook marketplace']
        
        for platform in platforms:
            platform_data = self._analyze_platform_presence(search_results, username, platform)
            results['platform_presence'][platform] = platform_data
            results['total_listings_found'] += platform_data.get('listings_count', 0)
            if platform_data.get('reviews'):
                results['review_summary'].extend(platform_data['reviews'])
        
        results['risk_level'] = self._calculate_user_risk_level(results)
        return results

    def get_user_listings(self, username: str) -> Dict:
        """Retrieves and analyzes user's listings from other platforms"""
        search_query = f"{username} real estate listings current"
        search_results = self.search.run(search_query)
        
        return {
            'username': username,
            'other_listings': self._extract_listings(search_results),
            'analysis': self._analyze_user_listing_patterns(search_results)
        }

    def validate_listing(self, listing_data: Dict) -> Dict:
        """Main method to validate a real estate listing"""
        # Add user verification if username is provided
        if 'username' in listing_data:
            user_verification = self.verify_user(listing_data['username'])
            listing_data['user_verification'] = user_verification
            
            # Get user's other listings
            user_listings = self.get_user_listings(listing_data['username'])
            listing_data['user_listings'] = user_listings
            
        # Add rating analysis if available
        if 'average_rating' in listing_data:
            rating_analysis = self.analyze_rating(listing_data['average_rating'])
            listing_data['rating_analysis'] = rating_analysis
            
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

    def analyze_rating(self, rating: float) -> Dict:
        """Analyzes the numerical rating and provides context"""
        rating_analysis = {
            'rating': rating,
            'risk_level': 'low' if rating >= 4.0 else 'medium' if rating >= 3.0 else 'high',
            'confidence_factor': min(rating / 5.0, 1.0),
            'recommendation': self._get_rating_recommendation(rating)
        }
        return rating_analysis

    def check_cross_platform(self, listing_data: Dict) -> Dict:
        """Checks for the same property listing across different platforms"""
        search_query = f"{listing_data['address']} {listing_data['price']} real estate listing"
        search_results = self.search.run(search_query)
        
        return {
            'cross_platform_listings': search_results,
            'platforms_found': self._extract_platforms(search_results)
        }

    def _analyze_platform_presence(self, search_results: str, username: str, platform: str) -> Dict:
        """Analyzes user's presence on a specific platform"""
        platform_data = {
            'listings_count': 0,
            'reviews': [],
            'verified_profile': False,
            'activity_level': 'unknown'
        }
        
        # Search for platform-specific information
        platform_search = self.search.run(f"{username} {platform} real estate agent reviews")
        
        # Extract information using LLM
        analysis_prompt = f"""
        Analyze the following search results for {username} on {platform} and extract:
        1. Approximate number of listings
        2. Any reviews or ratings
        3. Whether the profile appears verified
        4. Level of activity (high/medium/low)
        
        Search results: {platform_search}
        """
        
        try:
            result = self.llm.invoke(analysis_prompt)
            analysis = json.loads(result.content)
            platform_data.update(analysis)
        except Exception as e:
            platform_data['error'] = str(e)
            
        return platform_data

    def _calculate_user_risk_level(self, user_data: Dict) -> str:
        """Calculates user risk level based on various factors"""
        risk_factors = []
        
        # Check platform presence
        active_platforms = sum(1 for p in user_data['platform_presence'].values() 
                             if p.get('listings_count', 0) > 0)
        
        if active_platforms == 0:
            risk_factors.append('high')
        elif active_platforms == 1:
            risk_factors.append('medium')
        else:
            risk_factors.append('low')
            
        # Check total listings
        if user_data['total_listings_found'] == 0:
            risk_factors.append('high')
        elif user_data['total_listings_found'] < 3:
            risk_factors.append('medium')
        else:
            risk_factors.append('low')
            
        # Determine final risk level
        if 'high' in risk_factors:
            return 'high'
        elif 'medium' in risk_factors:
            return 'medium'
        return 'low'

    def _extract_listings(self, search_results: str) -> List[Dict]:
        """Extracts listing information from search results"""
        extract_prompt = """
        Extract listing information from the following search results. 
        Format as a list of listings with price, location, and date if available:
        
        {search_results}
        """
        
        result = self.llm.invoke(extract_prompt.format(search_results=search_results))
        try:
            return json.loads(result.content)
        except:
            return []

    def _analyze_user_listing_patterns(self, search_results: str) -> Dict:
        """Analyzes patterns in user's listing behavior"""
        analysis_prompt = """
        Analyze the following search results for patterns in user listing behavior.
        Look for:
        1. Price consistency
        2. Location patterns
        3. Listing frequency
        4. Any suspicious patterns
        
        Search results: {search_results}
        """
        
        result = self.llm.invoke(analysis_prompt.format(search_results=search_results))
        try:
            return json.loads(result.content)
        except:
            return {'error': 'Could not analyze listing patterns'}

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

    def _get_rating_recommendation(self, rating: float) -> str:
        """Provides recommendation based on rating"""
        if rating >= 4.5:
            return "Highly trusted listing with excellent reviews"
        elif rating >= 4.0:
            return "Well-reviewed listing, generally trustworthy"
        elif rating >= 3.0:
            return "Average rating - recommend additional verification"
        elif rating >= 2.0:
            return "Below average rating - exercise caution"
        else:
            return "Very low rating - significant red flags present"

    def _create_prompt(self) -> PromptTemplate:
        """Creates the agent prompt template"""
        return PromptTemplate.from_template(
            """You are a real estate safety expert AI agent. Your goal is to validate property 
            listings and identify potential fraud or suspicious activity.
            
            Analyze the given property listing using the available tools and provide a detailed 
            safety report including:
            1. Image validation results
            2. Sentiment analysis from reviews
            3. Cross-platform presence
            4. Review rating analysis
            5. User verification results
            6. Overall trust score
            7. Potential red flags
            
            Property listing: {input}
            
            Think through this step-by-step:
            1. First, validate the images
            2. Then, analyze any available reviews and ratings
            3. Check for cross-platform listings
            4. Verify the user's credibility
            5. Synthesize all information for a final assessment
            
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
        """Calculates trust score based on all factors"""
        factors = []
        
        # Rating factor
        if 'rating_analysis' in results:
            rating_factor = results['rating_analysis']['confidence_factor']
            factors.append(rating_factor)
        
        # User verification factor
        if 'user_verification' in results:
            user_risk = results['user_verification']['risk_level']
            user_factor = 1.0 if user_risk == 'low' else 0.5 if user_risk == 'medium' else 0.0
            factors.append(user_factor)
        
        return sum(factors) / len(factors) if factors else 0.0

    def _generate_summary(self, results: Dict) -> str:
        """Generates a human-readable summary of the validation results"""
        return str(results.get('output', 'No summary available'))

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generates actionable recommendations based on validation results"""
        recommendations = []
        
        # Add rating-based recommendation if available
        if 'rating_analysis' in results:
            recommendations.append(results['rating_analysis']['recommendation'])
        
        # Add user verification recommendations
        if 'user_verification' in results:
            risk_level = results['user_verification']['risk_level']
            if risk_level == 'high':
                recommendations.append("Exercise extreme caution - limited user history found")
            elif risk_level == 'medium':
                recommendations.append("Verify additional user information before proceeding")
            else:
                recommendations.append("User has established presence across platforms")
        
        # Add default recommendation if none available
        if not recommendations:
            recommendations.append("Insufficient data for specific recommendations")
            
        return recommendations

# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = RealEstateSafetyAgent(openai_api_key="your-api-key")

    # Example listing data
    listing_data = {
        "address": "123 Main St, City, State",
        "price": 500000,
        "description": "Beautiful 3-bedroom house...",
        "images": ["url1", "url2"],
        "reviews": ["Great property!", "The location is perfect"],
        "average_rating": 4.5,
        "username": "realtor_john_doe"
    }

    # Validate the listing
    results = agent.validate_listing(listing_data)
    print(json.dumps(results, indent=2))
