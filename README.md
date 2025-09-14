# estrategIA

Conectar por ssh al servidor jackson.rovit.ua.es en tres consolas diferentes:

1. Para poner en marcha Qdrant
     1. cd rmarti/prueba/estrategIA
     2. source estrategIA/bin/activate
     3. docker run -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
     4. Si da error el paso anterior. Ejecutar "docker ps" y con "docker stop nombre" cerrar el contenedor que ya esté lanzado para volverlo a lanzar docker run -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant

2. Poner en marcha la API
     1. cd rmarti/prueba/estrategIA
     2. source estrategIA/bin/activate
     3. cd src/
     4. python3 app.py
     5. uvicorn app:app --reload
  
3. Poner en marcha el front
    1. cd rmarti/prueba/chat-llm
    2. npm run dev
  
En otras tres nuevas ventanas de consola, redireccionar puertos desde el servidor a nuestro localhost:

1.   ssh -L 6333:localhost:6333 rmarti@jackson.rovit.ua.es (qdrant)
2.  ssh -L 3000:localhost:3000 rmarti@jackson.rovit.ua.es (front)
3.   ssh -L 8000:localhost:8000 rmarti@jackson.rovit.ua.es (api)

En el navegador local acceder a:

http://localhost:6333/dashboard#/welcome
http://localhost:8000/docs
http://localhost:3000/
