# 1. Usar uma imagem leve do Python
FROM python:3.12-slim

# 2. Definir o diretório de trabalho dentro do container
WORKDIR /app

# 3. Copiar apenas o arquivo de requisitos primeiro (melhora o cache do Docker)
COPY requirements.txt .

# 4. Instalar as dependências
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar o restante dos arquivos do projeto
COPY . .

# 6. Expor a porta que o FastAPI vai usar
EXPOSE 8000

# 7. Comando para rodar a API usando o Uvicorn
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]