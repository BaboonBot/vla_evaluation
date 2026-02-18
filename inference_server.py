"""
Inference Server for LeRobot Models

This server provides a REST API to serve LeRobot policy models for inference.
It supports multiple policy types including SmolVLA, ACT, Diffusion, and others.

Usage:
    python inference_server.py --model_id=NLTuan/smolvla_red_block_in_tape --port=8000

Environment Variables:
    MODEL_ID: Model identifier on HuggingFace Hub or local path
    PORT: Server port (default: 8000)
    HOST: Server host (default: 0.0.0.0)
    DEVICE: Device to run inference on (default: cuda if available, else cpu)
"""

import sys
sys.path.insert(0, "/workspace/lerobot/src")

import argparse
import base64
import io
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import get_policy_class


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================

class InferenceRequest(BaseModel):
    """Request model for inference endpoint."""
    observation: Dict[str, Any] = Field(
        ..., 
        description="Observation dictionary containing camera images and robot state"
    )
    action_steps: Optional[int] = Field(
        None,
        description="Number of action steps to predict (uses model default if not specified)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "observation": {
                    "observation.state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "observation.images.top": "base64_encoded_image_string"
                },
                "action_steps": 10
            }
        }


class InferenceResponse(BaseModel):
    """Response model for inference endpoint."""
    actions: List[List[float]] = Field(
        ...,
        description="Predicted action sequence as a list of action vectors"
    )
    action_dim: int = Field(..., description="Dimension of each action vector")
    num_steps: int = Field(..., description="Number of predicted action steps")
    model_info: Dict[str, Any] = Field(..., description="Model configuration info")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    model_loaded: bool
    model_id: Optional[str]
    device: str


class ModelInfo(BaseModel):
    """Response model for model info endpoint."""
    model_id: str
    policy_type: str
    action_dim: int
    chunk_size: int
    max_state_dim: int
    camera_inputs: List[str]
    device: str


# ============================================================================
# Inference Server
# ============================================================================

class LeRobotInferenceServer:
    """Inference server for LeRobot models."""
    
    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        policy_type: str = "auto"
    ):
        """
        Initialize the inference server.
        
        Args:
            model_id: Model identifier on HuggingFace Hub or local path
            device: Device to run inference on (cuda/cpu)
            policy_type: Policy type (auto, smolvla, act, diffusion, etc.)
        """
        self.model_id = model_id
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model and processors
        self.model = None
        self.preprocessor = None
        self.postprocessor = None
        self.policy_type = policy_type
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the model and preprocessors."""
        try:
            logger.info(f"Loading model from {self.model_id}...")
            
            # Auto-detect policy type from model config if needed
            if self.policy_type == "auto":
                # Try SmolVLA first as default
                try:
                    self.model = SmolVLAPolicy.from_pretrained(self.model_id)
                    self.policy_type = "smolvla"
                except Exception:
                    # Try to infer from model_id or use generic policy loader
                    logger.info("Failed to load as SmolVLA, will try other policy types")
                    raise
            else:
                # Load specific policy type
                policy_class = get_policy_class(self.policy_type)
                self.model = policy_class.from_pretrained(self.model_id)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully")
            logger.info(f"Policy type: {self.policy_type}")
            logger.info(f"Camera inputs: {list(self.model.config.image_features.keys())}")
            logger.info(f"Action dim: {self.model.config.action_feature.shape[0]}")
            logger.info(f"Chunk size: {self.model.config.n_action_steps}")
            logger.info(f"Max state dim: {self.model.config.max_state_dim}")
            
            # Load preprocessor and postprocessor
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                self.model.config,
                self.model_id,
                preprocessor_overrides={"device_processor": {"device": str(self.device)}},
            )
            logger.info("Preprocessors loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    @staticmethod
    def decode_base64_image(encoded_str: str) -> np.ndarray:
        """Decode base64 image string to numpy array in CHW format (to match dataset format)."""
        try:
            image_data = base64.b64decode(encoded_str)
            image = Image.open(io.BytesIO(image_data))
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Convert to numpy array in HWC format first
            img_array = np.array(image, dtype=np.uint8)
            # Transpose to CHW format (matches how dataset stores images)
            # (H, W, C) -> (C, H, W)
            img_chw = np.transpose(img_array, (2, 0, 1))
            return img_chw
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            raise
    
    def preprocess_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess observation by decoding base64 images and converting to torch tensors.
        
        Args:
            observation: Raw observation dictionary (may contain base64 encoded images)
            
        Returns:
            Processed observation with decoded images as tensors with batch dimension
        """
        processed = {}
        
        for key, value in observation.items():
            if 'images' in key and isinstance(value, str):
                # This is a base64 encoded image - decode and convert to tensor
                try:
                    img_array = self.decode_base64_image(value)  # Returns CHW format uint8
                    # Convert to tensor, normalize to [0, 1], and add batch dimension
                    # (C,H,W) uint8 -> (1,C,H,W) float32
                    img_tensor = torch.from_numpy(img_array).float() / 255.0
                    img_tensor = img_tensor.unsqueeze(0).to(self.device)
                    processed[key] = img_tensor
                except Exception as e:
                    logger.warning(f"Failed to decode image for key '{key}': {e}")
            elif isinstance(value, list):
                # Convert lists (like state) to tensors with batch dimension
                processed[key] = torch.tensor([value], dtype=torch.float32).to(self.device)
            elif isinstance(value, str):
                # Keep strings as-is (like task)
                processed[key] = [value]  # Wrap in list for batch
            else:
                processed[key] = value
        
        return processed
    
    def predict(
        self, 
        observation: Dict[str, Any],
        action_steps: Optional[int] = None
    ) -> np.ndarray:
        """
        Run inference on the given observation.
        
        Args:
            observation: Observation dictionary
            action_steps: Number of action steps to predict
            
        Returns:
            Predicted actions as numpy array
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Decode base64 images if present
            observation = self.preprocess_observation(observation)
            
            # Preprocess observation
            processed_obs = self.preprocessor(observation)
            
            # Run inference
            with torch.no_grad():
                if hasattr(self.model, 'predict_action_chunk'):
                    actions = self.model.predict_action_chunk(processed_obs)
                else:
                    # Generic prediction
                    actions = self.model.forward(processed_obs)
            
            # Postprocess if needed
            if self.postprocessor is not None:
                actions = self.postprocessor(actions)
            
            # Convert to numpy
            if isinstance(actions, torch.Tensor):
                actions = actions.cpu().numpy()
            
            return actions
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration information."""
        if self.model is None:
            return {}
        
        return {
            "model_id": self.model_id,
            "policy_type": self.policy_type,
            "action_dim": self.model.config.action_feature.shape[0],
            "chunk_size": self.model.config.n_action_steps,
            "max_state_dim": self.model.config.max_state_dim,
            "camera_inputs": list(self.model.config.image_features.keys()),
            "device": str(self.device)
        }


# ============================================================================
# FastAPI Application
# ============================================================================

# Global server instance
server: Optional[LeRobotInferenceServer] = None


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="LeRobot Inference Server",
        description="REST API for serving LeRobot policy models",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "LeRobot Inference Server",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "model_info": "/model/info",
                "inference": "/predict"
            }
        }
    
    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy" if server is not None and server.model is not None else "unhealthy",
            model_loaded=server is not None and server.model is not None,
            model_id=server.model_id if server is not None else None,
            device=str(server.device) if server is not None else "unknown"
        )
    
    @app.get("/model/info", response_model=ModelInfo)
    async def model_info():
        """Get model information."""
        if server is None or server.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        info = server.get_model_info()
        return ModelInfo(**info)
    
    @app.post("/predict", response_model=InferenceResponse)
    async def predict(request: InferenceRequest):
        """
        Run inference on the given observation.
        
        Args:
            request: Inference request with observation data
            
        Returns:
            Predicted actions
        """
        if server is None or server.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        try:
            # Run inference
            actions = server.predict(
                observation=request.observation,
                action_steps=request.action_steps
            )
            
            # Convert to list format
            if len(actions.shape) == 2:
                actions_list = actions.tolist()
            elif len(actions.shape) == 3:
                # Batch dimension present, take first item
                actions_list = actions[0].tolist()
            else:
                actions_list = [actions.tolist()]
            
            model_info = server.get_model_info()
            
            return InferenceResponse(
                actions=actions_list,
                action_dim=len(actions_list[0]) if actions_list else 0,
                num_steps=len(actions_list),
                model_info=model_info
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference failed: {str(e)}"
            )
    
    return app


# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LeRobot Inference Server")
    parser.add_argument(
        "--model_id",
        type=str,
        default=os.getenv("MODEL_ID", "NLTuan/smolvla_red_block_in_tape"),
        help="Model identifier on HuggingFace Hub or local path"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Server port"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("HOST", "0.0.0.0"),
        help="Server host"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.getenv("DEVICE"),
        help="Device to run inference on (cuda/cpu)"
    )
    parser.add_argument(
        "--policy_type",
        type=str,
        default=os.getenv("POLICY_TYPE", "auto"),
        help="Policy type (auto, smolvla, act, diffusion, etc.)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    # Initialize server
    global server
    logger.info("Initializing inference server...")
    server = LeRobotInferenceServer(
        model_id=args.model_id,
        device=args.device,
        policy_type=args.policy_type
    )
    
    # Create app
    app = create_app()
    
    # Run server
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
