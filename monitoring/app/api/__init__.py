from fastapi import APIRouter
from .endpoints import performance, aderencia  # importa os módulos dos endpoints

router = APIRouter()
router.include_router(performance.router, prefix="/performance")
router.include_router(aderencia.router, prefix="/aderencia")
