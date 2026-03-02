from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import register, recognize

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(register.router, prefix="/api", tags=["register"])
app.include_router(recognize.router, prefix="/api", tags=["recognize"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Face Recognition API"}