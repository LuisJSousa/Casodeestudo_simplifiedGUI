from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import List, Optional
import pandas as pd
import os
from datetime import datetime
import json
import httpx
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


app = FastAPI(title="Excel Classifier API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Only allow your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# Add request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    print(f"\n=== New Request ===")
    print(f"Method: {request.method}")
    print(f"URL: {request.url}")
    print(f"Headers: {request.headers}")
    
    response = await call_next(request)
    
    print(f"\n=== Response ===")
    print(f"Status Code: {response.status_code}")
    print(f"Headers: {response.headers}")
    
    return response

# Configuration
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:1234/v1/chat/completions")
UPLOAD_FOLDER = "uploads"

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class Item(BaseModel):
    article: str
    description: str
    quantity: float
    classification: Optional[str] = None  # Ex: "Chapter: ..., Subchapter: ..., Item: ..."
    unit_price: float = 0.0

async def classify_text(text: str) -> str:
    """Cria o prompt Alpaca-style e envia para o LLM"""
    try:
        # Instrução fixa para classificação
        instruction = (
            "Segue-se uma instrução que descreve uma tarefa, emparelhada com uma entrada que fornece mais contexto. "
            "Escreva uma resposta que complete adequadamente o pedido\n\n"
            "### Instruction:\n"
            "Classifica o input do user em Capítulo, Subcapítulo e Item.\n\n"
            "### Input:\n"
        )
        
        # Criar prompt com base na descrição
        full_prompt = f"{instruction}{text}\n\n### Response:"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                LLM_API_URL,
                json={
                    "model": "funsloth/unsloth.Q4_K_M-GGUF",  # Ou o nome que estiveres a usar
                    "messages": [
                        {"role": "system", "content": "Classifica o input do user em Capítulo, Subcapítulo e Item."},
                        {"role": "user", "content": full_prompt}
                    ],
                    "temperature": 0.01,
                    "max_tokens": 100,
                    "top_p": 0.95,
                    "stream": False
                },
                timeout=120.0
            )
            response.raise_for_status()
            result = response.json()
            output = result["choices"][0]["message"]["content"].strip()
            
            return output if output else "Classificação não disponível"

    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
        return f"HTTP Error: {e.response.status_code}"
    except Exception as e:
        print(f"Error calling LLM: {str(e)}")
        return "Erro na classificação"

@app.post("/api/upload", response_model=List[Item])
async def upload_file(file: UploadFile = File(...)):
    """Handle file upload and process with LLM"""
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only Excel files are allowed")
    
    try:
        # Reset file pointer to the beginning
        file.file.seek(0)
        
        try:
            # First, read the file to find the header row
            temp_df = pd.read_excel(file.file, header=None)
            
            # Find the row that contains 'Description' in any column
            header_row = None
            for idx, row in temp_df.iterrows():
                if any('Description' in str(cell) for cell in row.values if pd.notna(cell)):
                    header_row = idx
                    break
            
            # Reset file pointer again
            file.file.seek(0)
            
            if header_row is None:
                # If no header row found, use first row as data with default column names
                df = pd.read_excel(file.file, header=None)
                if len(df.columns) >= 3:
                    df.columns = ['Article', 'Description', 'Quantity'] + [f'Unnamed_{i}' for i in range(3, len(df.columns))]
            else:
                # Read the file with the found header row
                df = pd.read_excel(file.file, header=header_row)
                
                # Clean up column names
                df = df.rename(columns={
                    col: str(col).strip() for col in df.columns
                })
            
            # Print column names and sample data for debugging
            print("\nColumns in the uploaded file:", df.columns.tolist())
            print("\nFirst 10 rows of data:")
            print(df.head(10).to_string())
            
            # Check if we have the required columns
            required_columns = {'Article', 'Description', 'Quantity'}
            if not all(col in df.columns for col in required_columns):
                missing = required_columns - set(df.columns)
                raise ValueError(f"Missing required columns: {', '.join(missing)}")
                
            # Clean the data
            df = df.dropna(how='all')  # Drop completely empty rows
            df = df.fillna('')  # Replace remaining NaN with empty strings
            
            # Print all non-empty descriptions
            print("\nNon-empty descriptions found in the file:")
            non_empty_descriptions = df[df['Description'].astype(str).str.strip() != '']['Description']
            for idx, desc in enumerate(non_empty_descriptions, 1):
                print(f"{idx}. {str(desc).strip()}")
                
        except Exception as e:
            print(f"Error reading Excel file: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Error reading Excel file: {str(e)}. Please ensure the file has the correct format with 'Article', 'Description', and 'Quantity' columns."
            )
        
        # Process each non-empty row with LLM
        items = []
        for _, row in df.iterrows():
            description = str(row['Description']).strip()
            if not description or description.lower() == 'nan' or description.lower() == 'description':
                continue
                
            print(f"\nProcessing description: {description}")
            try:
                classification = await classify_text(description)
                print(f"Classification result: {classification}")
                
                item = Item(
                    article=str(row['Article']) if 'Article' in row and str(row['Article']).lower() != 'nan' else '',
                    description=description,
                    quantity=float(row['Quantity']) if 'Quantity' in row and str(row['Quantity']).strip() and str(row['Quantity']).lower() != 'nan' else 0,
                    classification=classification,
                    unit_price=float(row['Unit Price']) if 'Unit Price' in row and str(row['Unit Price']).strip() and str(row['Unit Price']).lower() != 'nan' else 0.0
                )
                items.append(item)
            except Exception as e:
                print(f"Error processing row: {e}")
                # Skip this row if there's an error
                continue
        
        return items
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/export")
async def export_to_excel(data: List[Item]):
    """Export processed data to Excel"""
    try:
        if not data:
            raise HTTPException(status_code=400, detail="No data to export")
            
        # Convert to DataFrame
        try:
            df = pd.DataFrame([item.dict() for item in data])
            
            # Ensure required columns exist
            required_columns = ['article', 'description', 'quantity', 'classification', 'unit_price']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = 0.0 if col == 'unit_price' else ''
            
            # Create output directory if it doesn't exist
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            
            # Create output file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"export_{timestamp}.xlsx"
            output_path = os.path.join(UPLOAD_FOLDER, output_filename)
            
            # Save to Excel
            df.to_excel(output_path, index=False, engine='openpyxl')
            
            # Verify file was created
            if not os.path.exists(output_path):
                raise Exception("Failed to create output file")
                
            # Return the file
            return FileResponse(
                output_path,
                filename=output_filename,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error creating Excel file: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during export: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
