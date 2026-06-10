#!/usr/bin/env python3
"""
Script to prepare multi-turn image editing dataset.
Creates LMDB database from directory structure or JSON metadata.
"""

import argparse
import json
import os
import lmdb
import pickle
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import io


def create_lmdb_from_directory(data_dir: str, output_path: str, num_turns: int = 3):
    """
    Create LMDB database from directory structure.
    
    Expected structure:
    data_dir/
        sample_0001/
            image_0.jpg
            image_1.jpg
            image_2.jpg
            prompts.txt
        sample_0002/
            image_0.jpg
            image_1.jpg
            image_2.jpg
            prompts.txt
    """
    data_dir = Path(data_dir)
    sample_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    print(f"Found {len(sample_dirs)} sample directories")
    
    # Create LMDB environment
    env = lmdb.open(output_path, map_size=1099511627776)  # 1TB
    
    with env.begin(write=True) as txn:
        # Store metadata
        metadata = {
            'num_samples': len(sample_dirs),
            'num_turns': num_turns,
            'num_images': num_turns,
            'num_prompts': num_turns - 1
        }
        txn.put(b'metadata', pickle.dumps(metadata))
        
        # Process each sample
        for idx, sample_dir in enumerate(tqdm(sample_dirs, desc="Processing samples")):
            try:
                # Load images
                images = []
                for i in range(num_turns):
                    img_path = sample_dir / f"image_{i}.jpg"
                    if not img_path.exists():
                        # Try other extensions
                        for ext in ['.png', '.jpeg', '.webp']:
                            img_path = sample_dir / f"image_{i}{ext}"
                            if img_path.exists():
                                break
                    
                    if img_path.exists():
                        with Image.open(img_path) as img:
                            img = img.convert('RGB')
                            # Convert to bytes
                            img_buffer = io.BytesIO()
                            img.save(img_buffer, format='PNG')
                            images.append(img_buffer.getvalue())
                    else:
                        print(f"Warning: Image {i} not found in {sample_dir}")
                        # Use a dummy image
                        dummy_img = Image.new('RGB', (512, 512), color='black')
                        img_buffer = io.BytesIO()
                        dummy_img.save(img_buffer, format='PNG')
                        images.append(img_buffer.getvalue())
                
                # Load prompts
                prompts = []
                prompts_file = sample_dir / "prompts.txt"
                if prompts_file.exists():
                    with open(prompts_file, 'r', encoding='utf-8') as f:
                        prompts = [line.strip() for line in f.readlines() if line.strip()]
                else:
                    # Try JSON format
                    prompts_file = sample_dir / "prompts.json"
                    if prompts_file.exists():
                        with open(prompts_file, 'r', encoding='utf-8') as f:
                            prompts = json.load(f)
                
                # Ensure we have the right number of prompts
                if len(prompts) < num_turns - 1:
                    # Pad with dummy prompts
                    while len(prompts) < num_turns - 1:
                        prompts.append(f"Edit step {len(prompts) + 1}")
                else:
                    # Truncate if too many
                    prompts = prompts[:num_turns - 1]
                
                # Store in LMDB
                for i, img_data in enumerate(images):
                    txn.put(f'image_{idx}_{i}'.encode(), img_data)
                
                for i, prompt in enumerate(prompts):
                    txn.put(f'prompt_{idx}_{i}'.encode(), prompt.encode('utf-8'))
                    
            except Exception as e:
                print(f"Error processing {sample_dir}: {e}")
                continue
    
    env.close()
    print(f"LMDB database created at {output_path}")


def create_lmdb_from_json(json_path: str, image_dir: str, output_path: str, num_turns: int = 3):
    """
    Create LMDB database from JSON metadata file.
    
    JSON format:
    [
        {
            "images": ["path/to/image_0.jpg", "path/to/image_1.jpg", "path/to/image_2.jpg"],
            "prompts": ["First edit instruction", "Second edit instruction"]
        },
        ...
    ]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    image_dir = Path(image_dir)
    
    print(f"Found {len(metadata)} samples in JSON")
    
    # Create LMDB environment
    env = lmdb.open(output_path, map_size=1099511627776)  # 1TB
    
    with env.begin(write=True) as txn:
        # Store metadata
        metadata_info = {
            'num_samples': len(metadata),
            'num_turns': num_turns,
            'num_images': num_turns,
            'num_prompts': num_turns - 1,
            'source': 'json'
        }
        txn.put(b'metadata', pickle.dumps(metadata_info))
        
        # Process each sample
        for idx, sample in enumerate(tqdm(metadata, desc="Processing samples")):
            try:
                # Load images
                images = []
                for img_path in sample['images'][:num_turns]:
                    full_path = image_dir / img_path
                    if full_path.exists():
                        with Image.open(full_path) as img:
                            img = img.convert('RGB')
                            # Convert to bytes
                            img_buffer = io.BytesIO()
                            img.save(img_buffer, format='PNG')
                            images.append(img_buffer.getvalue())
                    else:
                        print(f"Warning: Image not found: {full_path}")
                        # Use a dummy image
                        dummy_img = Image.new('RGB', (512, 512), color='black')
                        img_buffer = io.BytesIO()
                        dummy_img.save(img_buffer, format='PNG')
                        images.append(img_buffer.getvalue())
                
                # Load prompts
                prompts = sample.get('prompts', [])[:num_turns - 1]
                
                # Ensure we have the right number of prompts
                if len(prompts) < num_turns - 1:
                    # Pad with dummy prompts
                    while len(prompts) < num_turns - 1:
                        prompts.append(f"Edit step {len(prompts) + 1}")
                
                # Store in LMDB
                for i, img_data in enumerate(images):
                    txn.put(f'image_{idx}_{i}'.encode(), img_data)
                
                for i, prompt in enumerate(prompts):
                    txn.put(f'prompt_{idx}_{i}'.encode(), prompt.encode('utf-8'))
                    
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
    
    env.close()
    print(f"LMDB database created at {output_path}")


def verify_lmdb(lmdb_path: str, num_samples: int = 5):
    """Verify LMDB database by reading a few samples."""
    env = lmdb.open(lmdb_path, readonly=True)
    
    with env.begin() as txn:
        # Read metadata
        metadata_data = txn.get(b'metadata')
        if metadata_data:
            metadata = pickle.loads(metadata_data)
            print(f"Metadata: {metadata}")
        
        # Read a few samples
        for i in range(min(num_samples, metadata.get('num_samples', 0))):
            print(f"\nSample {i}:")
            
            # Read images
            for j in range(metadata['num_images']):
                img_key = f'image_{i}_{j}'.encode()
                img_data = txn.get(img_key)
                if img_data:
                    print(f"  Image {j}: {len(img_data)} bytes")
                else:
                    print(f"  Image {j}: Not found")
            
            # Read prompts
            for j in range(metadata['num_prompts']):
                prompt_key = f'prompt_{i}_{j}'.encode()
                prompt_data = txn.get(prompt_key)
                if prompt_data:
                    prompt = prompt_data.decode('utf-8')
                    print(f"  Prompt {j}: {prompt}")
                else:
                    print(f"  Prompt {j}: Not found")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Prepare multi-turn image editing dataset")
    parser.add_argument("--input_type", choices=["directory", "json"], required=True,
                       help="Input data type")
    parser.add_argument("--input_path", required=True,
                       help="Path to input directory or JSON file")
    parser.add_argument("--image_dir", 
                       help="Path to image directory (required for JSON input)")
    parser.add_argument("--output_path", required=True,
                       help="Path to output LMDB file")
    parser.add_argument("--num_turns", type=int, default=3,
                       help="Number of editing turns (default: 3)")
    parser.add_argument("--verify", action="store_true",
                       help="Verify created LMDB database")
    
    args = parser.parse_args()
    
    if args.input_type == "json" and not args.image_dir:
        parser.error("--image_dir is required when input_type is json")
    
    # Create LMDB database
    if args.input_type == "directory":
        create_lmdb_from_directory(args.input_path, args.output_path, args.num_turns)
    else:
        create_lmdb_from_json(args.input_path, args.image_dir, args.output_path, args.num_turns)
    
    # Verify if requested
    if args.verify:
        print("\nVerifying LMDB database...")
        verify_lmdb(args.output_path)


if __name__ == "__main__":
    main()