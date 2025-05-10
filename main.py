from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import dyscalculia, dysgraphia, dyslexia, dyscalculia_arithmetic,dyslexia_reading, dyslexia_letter_confusion
from routers import dysgraphia_tracing
from routers.dyslexia_spelling import router as spelling_router
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

# Serve static files for audio
app.mount("/audio", StaticFiles(directory=os.path.join(os.getcwd(), "audio/correct")), name="audio")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dusgraphia Test
app.include_router(dysgraphia.router, prefix="/dysgraphia", tags=["Dysgraphia Handwriting"])
app.include_router(dysgraphia_tracing.router, prefix="/dysgraphia_tracing", tags=["Dysgraphia Tracing"])

#Dyslexia Test
app.include_router(dyslexia.router, prefix="/dyslexia")
app.include_router(dyslexia_reading.router, prefix="/dyslexia_reading", tags=["Dyslexia Reading"])
app.include_router(spelling_router, prefix="/dyslexia/spelling", tags=["Dyslexia Spelling"])
app.include_router(dyslexia_letter_confusion.router, prefix="/dyslexia_letter_confusion", tags=["Dyslexia Letter Confusion"])
    

# Dyscalculia test
app.include_router(dyscalculia.router, prefix="/dyscalculia", tags=["Dyscalculua"])
app.include_router(dyscalculia_arithmetic.router, prefix="/dyscalculia_arithmetic", tags=["Dyscalculia Arithmetic"])

# Root endpoint
@app.get("/")
def root():
    return {"message": "EarlyEdge API is running!"}
