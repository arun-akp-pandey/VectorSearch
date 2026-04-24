import VectorContext

#print(VectorContext.add_images_from_directory())

#print(*(OptimizedEncoder.get_all_entities()), sep='\n')

#print(f"Collection deleted: {VectorContext.drop_collection()}")

print(VectorContext.search_by_image("Argument/los.jpg"))
