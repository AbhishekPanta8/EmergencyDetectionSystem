import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_template_variations(template, variations, placeholders):
    """Generate variations of a template message"""
    results = []
    for _ in range(variations):
        message = template
        for key, values in placeholders.items():
            message = message.replace(key, random.choice(values))
        results.append(message)
    return results

def create_large_emergency_dataset():
    # Common placeholders for variation
    locations = ['downtown', 'north district', 'south side', 'west end', 'east area', 
                'central park', 'riverside', 'industrial zone', 'suburban area', 
                'city center', 'metropolitan area', 'residential district', 'commercial zone',
                'highway 101', 'interstate 95', 'route 27', 'main street', 'harbor area']
    
    numbers = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '12', '15', '20', '25', '30']
    
    times = ['immediately', 'within next hour', 'in next 30 minutes', 'urgently',
             'as soon as possible', 'now', 'without delay']
    
    weather_conditions = ['heavy rain', 'strong winds', 'severe storm', 'lightning',
                         'hail', 'snow', 'ice', 'extreme heat', 'dense fog']
    
    magnitudes = ['severe', 'major', 'significant', 'massive', 'extensive', 'considerable',
                 'substantial', 'critical', 'serious', 'devastating']

    # Template messages with placeholders for each category
    critical_templates = [
        "URGENT: {MAGNITUDE} flooding in {LOCATION}, {NUMBER} people trapped! Rescue needed {TIME}! #emergency",
        "{NUMBER} casualties reported in {MAGNITUDE} fire near {LOCATION}. Evacuate {TIME}! #wildfire",
        "ALERT: {MAGNITUDE} tornado approaching {LOCATION}. {NUMBER} minutes to seek shelter! #tornado",
        "Building collapsed in {LOCATION} after {MAGNITUDE} earthquake. {NUMBER}+ people trapped. Emergency services responding.",
        "{MAGNITUDE} hurricane with {WEATHER} approaching {LOCATION}. Wind speeds {NUMBER}0mph. Immediate evacuation ordered!",
        "CRITICAL: {MAGNITUDE} gas leak detected in {LOCATION}. {NUMBER} block radius being evacuated {TIME}.",
        "{MAGNITUDE} accident on {LOCATION}: {NUMBER} vehicles involved. Multiple injuries. Avoid area {TIME}.",
        "EMERGENCY: {MAGNITUDE} chemical spill in {LOCATION}. {NUMBER} mile evacuation zone. Hazmat responding.",
        "Active shooter at {LOCATION}. {NUMBER} casualties reported. Police operation ongoing. Stay away!",
        "Dam failure imminent at {LOCATION}! {NUMBER} communities at risk. Evacuate {TIME}! #flood"
    ]

    high_priority_templates = [
        "{LOCATION} roads flooded due to {WEATHER}. {NUMBER} feet of water reported.",
        "Power outage affecting {NUMBER}k residents in {LOCATION} after {WEATHER}.",
        "{MAGNITUDE} brush fire near {LOCATION}, spreading {TIME}. {NUMBER} acres affected.",
        "{NUMBER} aftershocks reported in {LOCATION}. Magnitude {MAGNITUDE} on Richter scale.",
        "Hospital in {LOCATION} at {NUMBER}% capacity. Seeking additional resources.",
        "Water contamination alert for {LOCATION}. Affects {NUMBER}k residents. Boil advisory issued.",
        "Missing person in {LOCATION}: Last seen {NUMBER} hours ago during {WEATHER}.",
        "{NUMBER} traffic signals out in {LOCATION} due to {WEATHER}.",
        "Evacuation center at {LOCATION} reaching capacity. {NUMBER} people sheltered.",
        "{MAGNITUDE} landslide reported near {LOCATION}. {NUMBER} homes at risk."
    ]

    medium_priority_templates = [
        "Minor flooding in {LOCATION}. Water level risen {NUMBER} inches due to {WEATHER}.",
        "{WEATHER} forecast for {LOCATION}. Expected duration: {NUMBER} hours.",
        "Small fire contained in {LOCATION}. {NUMBER} acres affected. No evacuations needed.",
        "Traffic delay on {LOCATION}: {NUMBER} minute backup due to {MAGNITUDE} accident.",
        "Planned power outage in {LOCATION} for {NUMBER} hours tomorrow.",
        "Wildlife alert: {MAGNITUDE} bear sighted in {LOCATION}. Stay alert.",
        "Construction incident at {LOCATION}. Area secured, {NUMBER} workers affected.",
        "Water pressure issues reported in {LOCATION}. {NUMBER} households affected.",
        "Protest scheduled at {LOCATION}. Expecting {NUMBER}00 participants.",
        "Beach hazard at {LOCATION}: {WEATHER} causing dangerous conditions."
    ]

    placeholders = {
        '{LOCATION}': locations,
        '{NUMBER}': numbers,
        '{TIME}': times,
        '{WEATHER}': weather_conditions,
        '{MAGNITUDE}': magnitudes
    }

    # Generate variations for each template
    num_variations = 100  # This will create about 1000 critical, 1000 high priority, etc.

    data = {
        'text': [],
        'emergency_level': [],
        'sentiment': [],
        'is_emergency': [],
        'disaster_type': [],
        'source': [],
        'verified': [],
        'timestamp': []
    }

    # Generate data for each template category
    for templates, level, sentiment, is_emergency in [
        (critical_templates, 'CRITICAL', 'PANIC', True),
        (high_priority_templates, 'HIGH', 'NEGATIVE', True),
        (medium_priority_templates, 'MEDIUM', 'NEUTRAL', True)
    ]:
        for template in templates:
            variations = generate_template_variations(template, num_variations, placeholders)
            data['text'].extend(variations)
            data['emergency_level'].extend([level] * len(variations))
            data['sentiment'].extend([sentiment] * len(variations))
            data['is_emergency'].extend([is_emergency] * len(variations))
            
            # Generate corresponding disaster types
            if 'flood' in template.lower():
                disaster_type = 'FLOOD'
            elif 'fire' in template.lower():
                disaster_type = 'FIRE'
            elif 'tornado' in template.lower():
                disaster_type = 'TORNADO'
            elif 'earthquake' in template.lower():
                disaster_type = 'EARTHQUAKE'
            elif 'hurricane' in template.lower():
                disaster_type = 'HURRICANE'
            elif 'chemical' in template.lower():
                disaster_type = 'HAZMAT'
            elif 'shooter' in template.lower():
                disaster_type = 'VIOLENCE'
            else:
                disaster_type = 'OTHER'
            
            data['disaster_type'].extend([disaster_type] * len(variations))
            
            # Generate metadata
            data['source'].extend(np.random.choice(
                ['twitter', 'emergency_service', 'news_agency', 'public'],
                size=len(variations)
            ))
            data['verified'].extend(np.random.choice(
                [True, False],
                size=len(variations),
                p=[0.7, 0.3]
            ))
            
            # Generate timestamps over last 30 days
            start_date = datetime.now() - timedelta(days=30)
            timestamps = [start_date + timedelta(
                minutes=random.randint(0, 43200)
            ) for _ in range(len(variations))]
            data['timestamp'].extend(timestamps)

    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some noise to prevent exact duplicates
    df['text'] = df['text'].apply(lambda x: x + f" ID:{random.randint(1000, 9999)}")
    
    # Add low priority and non-emergency messages
    num_additional = len(df) // 4  # 25% more entries
    
    # Add low priority messages
    low_priority_df = pd.DataFrame({
        'text': [f"Minor incident: {random.choice(weather_conditions)} affecting {random.choice(locations)}. Use caution. #{random.randint(1000, 9999)}" for _ in range(num_additional)],
        'emergency_level': ['LOW'] * num_additional,
        'sentiment': ['NEUTRAL'] * num_additional,
        'is_emergency': [False] * num_additional,
        'disaster_type': ['WEATHER'] * num_additional,
        'source': np.random.choice(['twitter', 'public', 'news_agency'], size=num_additional),
        'verified': np.random.choice([True, False], size=num_additional, p=[0.5, 0.5]),
        'timestamp': [datetime.now() - timedelta(minutes=random.randint(0, 43200)) for _ in range(num_additional)]
    })
    
    # Add non-emergency messages
    non_emergency_df = pd.DataFrame({
        'text': [f"Regular update: Community event in {random.choice(locations)}. #{random.randint(1000, 9999)}" for _ in range(num_additional)],
        'emergency_level': ['NON_EMERGENCY'] * num_additional,
        'sentiment': ['POSITIVE'] * num_additional,
        'is_emergency': [False] * num_additional,
        'disaster_type': ['NONE'] * num_additional,
        'source': np.random.choice(['twitter', 'public', 'news_agency'], size=num_additional),
        'verified': np.random.choice([True, False], size=num_additional, p=[0.5, 0.5]),
        'timestamp': [datetime.now() - timedelta(minutes=random.randint(0, 43200)) for _ in range(num_additional)]
    })

    # Combine all DataFrames
    final_df = pd.concat([df, low_priority_df, non_emergency_df], ignore_index=True)
    
    # Shuffle the dataset
    final_df = final_df.sample(frac=1).reset_index(drop=True)
    
    return final_df

# Create and save the dataset
df = create_large_emergency_dataset()
df.to_csv('data/raw/large_emergency_dataset.csv', index=False)

print(f"Created dataset with {len(df)} entries")