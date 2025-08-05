# FastAPI Code Generation API with DeepSeek (Free LLM)
# Modified version of your structure to work with DeepSeek via OpenRouter

import os
import json
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv  # ✅ Add this

load_dotenv()  # ✅ And this line


import requests
import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =============================================================================
# MODELS / SCHEMAS
# =============================================================================

class CodeGenerationRequest(BaseModel):
    """Request model for code generation"""
    json_payload: Dict[Any, Any] = Field(..., description="Input JSON data")
    instructions: str = Field(..., description="What to generate")
    output_format: str = Field(default="xml", description="Output format (xml, python, javascript, sql, etc.)")
    model: str = Field(default="deepseek/deepseek-chat-v3-0324:free", description="LLM model to use")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Temperature for generation")
    max_tokens: int = Field(default=1500, ge=100, le=4000, description="Maximum tokens to generate")

class CodeGenerationResult(BaseModel):
    """Result model for code generation"""
    success: bool
    generated_code: Optional[str] = None
    code_type: str
    explanation: Optional[str] = None
    usage_example: Optional[str] = None
    additional_notes: Optional[str] = None
    error: Optional[str] = None
    generation_time: float
    model_used: str
    timestamp: datetime

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    api_available: bool

# =============================================================================
# LLM INTERFACE
# =============================================================================

class DeepSeekCodeGenerator:
    """DeepSeek LLM interface using OpenRouter"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "FastAPI Code Generator"
        }
    
    async def generate_code_async(self, request: CodeGenerationRequest) -> Dict[str, Any]:
        """Generate code asynchronously"""
        
        # Convert payload to JSON string
        payload_str = json.dumps(request.json_payload, indent=2)
        
        # Create system prompt
        system_prompt = f"""You are an expert code generator. Analyze JSON data and generate code based on instructions.

CAPABILITIES:
- Transform JSON to any format (XML templates, Json)
- Handle nested objects, arrays, conditional logic
- Generate template engines (Velocity, Freemarker, etc.)
- Create API responses, database schemas, validation code

ALWAYS return your response in this exact JSON format:
{{
  "generated_code": "your generated code here",
  "code_type": "{request.output_format}",
  "explanation": "brief explanation of what the code does",
  "usage_example": "how to use this code",
  "additional_notes": "any important notes"
}}"""

        user_prompt = f"""ANALYZE THIS JSON PAYLOAD:
{payload_str}

INSTRUCTIONS: {request.instructions}

OUTPUT FORMAT: {request.output_format}

Generate the requested code and return it in the JSON format specified."""

        # Prepare API payload
        api_payload = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens
        }
        
        # Make async API call
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    self.base_url,
                    headers=self.headers,
                    json=api_payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API call failed: {response.status} - {error_text}")
                    
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    # Try to parse as JSON
                    try:
                        parsed_result = json.loads(content)
                        return parsed_result
                    except:
                        # If not valid JSON, wrap it
                        return {
                            "generated_code": content,
                            "code_type": request.output_format,
                            "explanation": "Generated code based on your JSON payload and instructions",
                            "usage_example": "Use the generated_code field",
                            "additional_notes": "Raw response - may need formatting"
                        }
                        
            except Exception as e:
                raise Exception(f"Code generation failed: {str(e)}")
    
    def test_connection(self) -> bool:
        """Test if the API connection is working"""
        try:
            test_payload = {
                "model": "deepseek/deepseek-chat-v3-0324:free",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=test_payload,
                timeout=10
            )
            
            return response.status_code == 200
        except:
            return False

# =============================================================================
# CODE VALIDATOR
# =============================================================================

class CodeValidator:
    """Validate generated code"""
    
    @staticmethod
    def validate_xml(code: str) -> bool:
        """Basic XML validation"""
        try:
            import xml.etree.ElementTree as ET
            ET.fromstring(f"<root>{code}</root>")
            return True
        except:
            return code.strip().startswith('<') and code.strip().endswith('>')
    
    @staticmethod
    def validate_json(code: str) -> bool:
        """Validate JSON code"""
        try:
            json.loads(code)
            return True
        except:
            return False
    
    @staticmethod
    def validate_python(code: str) -> bool:
        """Basic Python syntax validation"""
        try:
            compile(code, '<string>', 'exec')
            return True
        except:
            return False
    
    def validate_code(self, code: str, code_type: str) -> Dict[str, Any]:
        """Validate generated code"""
        if not code:
            return {"valid": False, "message": "No code generated"}
        
        validation_methods = {
            "xml": self.validate_xml,
            "json": self.validate_json,
            "python": self.validate_python
        }
        
        if code_type in validation_methods:
            is_valid = validation_methods[code_type](code)
            return {
                "valid": is_valid,
                "message": "Code is valid" if is_valid else f"Invalid {code_type} syntax"
            }
        else:
            # For other formats, just check if code exists
            return {
                "valid": len(code.strip()) > 0,
                "message": "Basic validation passed"
            }

# =============================================================================
# SERVICE LAYER
# =============================================================================

class CodeGenerationService:
    """Main service for code generation"""
    
    def __init__(self, llm_interface: DeepSeekCodeGenerator, validator: CodeValidator):
        self.llm = llm_interface
        self.validator = validator
    
    async def generate_code(self, request: CodeGenerationRequest) -> CodeGenerationResult:
        """Generate code with validation"""
        start_time = datetime.now()
        
        try:
            # Generate code using LLM
            llm_result = await self.llm.generate_code_async(request)
            
            generated_code = llm_result.get("generated_code")
            
            # Validate the generated code
            validation_result = self.validator.validate_code(generated_code, request.output_format)
            
            # Calculate generation time
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return CodeGenerationResult(
                success=True,
                generated_code=generated_code,
                code_type=llm_result.get("code_type", request.output_format),
                explanation=llm_result.get("explanation"),
                usage_example=llm_result.get("usage_example"),
                additional_notes=f"Validation: {validation_result['message']}. {llm_result.get('additional_notes', '')}",
                generation_time=generation_time,
                model_used=request.model,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return CodeGenerationResult(
                success=False,
                code_type=request.output_format,
                error=str(e),
                generation_time=generation_time,
                model_used=request.model,
                timestamp=datetime.now()
            )

# =============================================================================
# FASTAPI APP
# =============================================================================

# Create the FastAPI app
app = FastAPI(
    title="DeepSeek Code Generation API",
    description="API for generating code from JSON specifications using DeepSeek LLM (Free)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
def get_api_key():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")
    return api_key

try:
    llm = DeepSeekCodeGenerator(api_key=get_api_key())
    validator = CodeValidator()
    service = CodeGenerationService(llm_interface=llm, validator=validator)
except ValueError as e:
    print(f"⚠️ Warning: {e}")
    print("Set environment variable: export OPENROUTER_API_KEY='your-key'")
    # Create dummy components for development
    llm = None
    validator = CodeValidator()
    service = None

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.post("/generate", response_model=CodeGenerationResult)
async def generate_code_endpoint(request: CodeGenerationRequest):
    """Generate code from JSON payload and instructions"""
    if service is None:
        raise HTTPException(
            status_code=500, 
            detail="Service not initialized. Please set OPENROUTER_API_KEY environment variable."
        )
    
    try:
        result = await service.generate_code(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    api_available = llm.test_connection() if llm else False
    
    return HealthResponse(
        status="healthy" if api_available else "degraded",
        timestamp=datetime.now(),
        api_available=api_available
    )

@app.get("/models")
async def get_available_models():
    """Get list of available models"""
    return {
        "models": [
            "deepseek/deepseek-r1:free",
            "deepseek/deepseek-chat-v3-0324:free", 
            "tngtech/deepseek-r1t-chimera:free"
        ],
        "default": "deepseek/deepseek-chat-v3-0324:free",
        "note": "All models are free via OpenRouter"
    }

@app.post("/generate/xml")
async def generate_xml_template(
    json_payload: Dict[Any, Any],
    instructions: str = "Generate XML transformation template"
):
    """Specialized endpoint for XML template generation (your original use case)"""
    request = CodeGenerationRequest(
        json_payload=json_payload,
        instructions=f"{instructions}. Use <json:object>, <json:property>, <Core:if>, <Core:forEach> syntax.",
        output_format="xml"
    )
    
    return await generate_code_endpoint(request)



# =============================================================================
# STARTUP/SHUTDOWN EVENTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup event"""
    print("🚀 DeepSeek Code Generation API starting up...")
    if llm and llm.test_connection():
        print("✅ DeepSeek API connection successful")
    else:
        print("⚠️ DeepSeek API connection failed - check your OPENROUTER_API_KEY")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event"""
    print("🔄 API shutting down...")

# =============================================================================
# MAIN / DEVELOPMENT SERVER
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",  # Change this to your filename
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )