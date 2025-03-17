import weaviate
from weaviate.classes.config import Configure
import os
from tqdm import tqdm

headers = {
    "X-OpenAI-Api-Key": os.getenv("OPENAI_APIKEY"),
    "X-Cohere-Api-Key": os.getenv("COHERE_APIKEY")
}  # Replace with your OpenAI and Cohere API key

client = weaviate.connect_to_local(
    port=80,            # The default REST port is 8080
    # grpc_port=50051   # Not needed, as the default gRPC port is 50051
)

print(client.is_ready())


source_objects = [
    {"title": "The Shawshank Redemption", "description": "A wrongfully imprisoned man forms an inspiring friendship while finding hope and redemption in the darkest of places."},
    {"title": "The Godfather", "description": "A powerful mafia family struggles to balance loyalty, power, and betrayal in this iconic crime saga."},
    {"title": "The Dark Knight", "description": "Batman faces his greatest challenge as he battles the chaos unleashed by the Joker in Gotham City."},
    {"title": "Jingle All the Way", "description": "A desperate father goes to hilarious lengths to secure the season's hottest toy for his son on Christmas Eve."},
    {"title": "A Christmas Carol", "description": "A miserly old man is transformed after being visited by three ghosts on Christmas Eve in this timeless tale of redemption."}
]

def batch_import_data(client, collection_name, source_objects, batch_size=100, error_threshold=10):
    """
    Batch import data into a Weaviate collection.
    
    Args:
        client: Weaviate client instance
        collection_name (str): Name of the collection to import to
        source_objects (list): List of dictionaries containing objects to import
        batch_size (int): Size of each batch (default 100)
        error_threshold (int): Maximum number of errors before stopping (default 10)
    
    Returns:
        tuple: (success_count, failed_objects)
    """
    try:
        # Get the collection
        collection = client.collections.get(collection_name)
        success_count = 0
        
        # Configure batch settings
        with collection.batch.dynamic() as batch:
            for src_obj in source_objects:
                try:
                    # Add object to batch
                    batch.add_object(
                        properties={
                            "title": src_obj["title"],
                            "description": src_obj["description"],
                            # Add other properties as needed
                        }
                    )
                    success_count += 1
                    
                    # Check for excessive errors
                    if batch.number_errors > error_threshold:
                        print(f"Batch import stopped after {success_count} objects due to excessive errors.")
                        break
                        
                except Exception as e:
                    print(f"Error adding object: {str(e)}")
                    print(f"Problematic object: {src_obj}")
                    continue
        
        # Get failed objects after batch completion
        failed_objects = collection.batch.failed_objects
        
        # Print summary
        print(f"Import completed:")
        print(f"Successfully imported: {success_count} objects")
        if failed_objects:
            print(f"Failed imports: {len(failed_objects)}")
            print(f"First failed object: {failed_objects[0]}")
        
        return success_count, failed_objects
        
    except Exception as e:
        print(f"Error during batch import: {str(e)}")
        return 0, []



def query_collection(client, collection_name, query_type, query, limit=2, additional_params=None):
    """
    Query a Weaviate collection using different query types.
    
    Args:
        client: Weaviate client instance
        collection_name (str): Name of the collection to query
        query_type (str): Type of query ('hybrid', 'near_text')
        query (str): The query string
        limit (int): Maximum number of results to return (default 2)
        additional_params (dict): Additional parameters for the query (optional)
    
    Returns:
        list: List of results with their properties
    """
    try:
        # Get the collection
        collection = client.collections.get(collection_name)
        
        # Initialize results list
        results = []
        
        # Set up base query parameters
        query_params = {
            "query": query,
            "limit": limit
        }
        
        # Add any additional parameters if provided
        if additional_params:
            query_params.update(additional_params)
        
        # Execute query based on query_type
        if query_type.lower() == 'hybrid':
            response = collection.query.hybrid(**query_params)
        elif query_type.lower() == 'near_text':
            response = collection.query.near_text(**query_params)
        else:
            raise ValueError(f"Unsupported query type: {query_type}")
        
        # Process and print results
        if hasattr(response, 'objects'):
            print(f"Found {len(response.objects)} results:")
            print("-"*30)
            
            for i, obj in enumerate(response.objects, 1):
                results.append(obj.properties)
                
                # Print result details
                print(f"Result {i}:")
                for key, value in obj.properties.items():
                    # Truncate long values for better display
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    print(f"{key}: {value}")
                
                # Print score if available
                if hasattr(obj, 'score'):
                    print(f"Score: {obj.score}")
                
                print("-"*30)
            
            if not response.objects:
                print("No results found.")
                
        return results

    except Exception as e:
        print(f"Error during query: {str(e)}")
        return []



def migrate_data(collection_src, collection_tgt):
    """
    Migration with explicit vector checking.
    """
    try:
        print(f"\nStarting migration")
        print(f"From: {collection_src.name} -> To: {collection_tgt.name}\n")

        # Get first object to check vector structure
        first_obj = next(collection_src.iterator(include_vector=True))
        print(f"Vector structure in first object: {first_obj.vector}")
        vector_keys = list(first_obj.vector.keys())
        if not vector_keys:
            raise ValueError("No vectors found in source collection")
        
        vector_key = vector_keys[0]
        print(f"Using vector key: {vector_key}\n")

        success_count = 0
        error_count = 0
        
        with collection_tgt.batch.fixed_size(batch_size=100) as batch:
            for q in tqdm(collection_src.iterator(include_vector=True)):
                try:
                    batch.add_object(
                        properties=q.properties,
                        vector=q.vector[vector_key],
                        uuid=q.uuid
                    )
                    success_count += 1
                    
                    if success_count % 100 == 0:
                        print(f"Processed objects - Success: {success_count}, Errors: {error_count}")
                        
                except Exception as e:
                    error_count += 1
                    print(f"Error processing object {q.uuid}: {str(e)}")
                    continue

        print("\nMigration Summary:")
        print(f"Successful transfers: {success_count}")
        print(f"Failed transfers: {error_count}")
        
        return True if error_count == 0 else False

    except Exception as e:
        print(f"\nFatal error during migration: {str(e)}")
        return False

#Wipe all data in weaviate DB
#client.collections.delete_all()

try:
    # Get list of all collections
    collections = client.collections.list_all()
    collection_names = collections.keys()

    # Check if OriginalCollection exists
    if "OriginalCollection" not in collection_names:
        print("Creating new OriginalCollection...")
        client.collections.create(
            "OriginalCollection",
            replication_config=Configure.replication(
                factor=3
            ),
            vectorizer_config=[
                Configure.NamedVectors.text2vec_openai(
                    name="title_vector",
                    source_properties=["title"],
                    model="text-embedding-3-large",
                    dimensions=1024
                )
            ],
            generative_config=Configure.Generative.openai(),
        )
        print("OriginalCollection created successfully")
        batch_import_data(client, "OriginalCollection", source_objects)
    else:
        print("OriginalCollection already exists")
        collection = client.collections.get("OriginalCollection")

        print("\nOPEN AI Model Processing: ")
        
        print("\nNear Text Search Processing: ")
        query_collection(client, "OriginalCollection", "near_text", "Christmas")

        print("\nHybrid Search Processing: ")
        query_collection(client, "OriginalCollection", "hybrid", "Family friendly")
        
        print("RAG Results: ")
        response = collection.generate.near_text(
            query="A movie",  # The model provider integration will automatically vectorize the query
            single_prompt="Categorize genre: {title}",
            limit=2
        )

        for obj in response.objects:
            print(obj.properties["title"])
            print(f"Generated output: {obj.generated}")  # Note that the generated output is per object

    # Check if NewCollection exists
    if "NewCollection" not in collection_names:
        print("Creating NewCollection...")
        client.collections.create(
            "NewCollection",
            replication_config=Configure.replication(
                factor=3
            ),
            vectorizer_config=[
                Configure.NamedVectors.text2vec_cohere(
                    name="title_vector",
                    source_properties=["title, description"],
                    model="embed-english-v3.0"
                )
            ],
            generative_config=Configure.Generative.cohere(),
        )
        print("NewCollection created successfully")
        #print(client.collections.get("NewCollection"))
    else:
        print("\nNewCollection already exists")
        #collection = client.collections.get("NewCollection")

        data_src = client.collections.get("OriginalCollection")
        data_tgt = client.collections.get("NewCollection")
        #print(data_tgt)
        
        # One-line check before migration
        if next(data_tgt.iterator(), None) is not None:
            print(f"Target collection '{data_tgt.name}' is not empty.")
        else:
            print(f"Target collection '{data_tgt.name}' is empty. Proceeding with migration...")
            migrate_data(data_src, data_tgt)

        print("\nCohere Model Processing: ")
        print("\nNear Text Search Processing: ")
        query_collection(client, "NewCollection", "near_text", "Christmas")

        print("\nHybrid Search Processing: ")
        query_collection(client, "NewCollection", "hybrid", "Family friendly")
        
        print("RAG results: ")
        response = collection.generate.near_text(
            query="A movie",  # The model provider integration will automatically vectorize the query
            single_prompt="Categorize genre: {title}",
            limit=2
        )

        for obj in response.objects:
            print(obj.properties["title"])
            print(f"Generated output: {obj.generated}")  # Note that the generated output is per object

except Exception as e:
    print(f"Error: {e}")
finally:
    client.close()
