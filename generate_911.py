import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_911_call_variations(template, variations, placeholders):
    """Generate variations of a 911 call template message."""
    results = []
    for _ in range(variations):
        call_text = template
        for key, values in placeholders.items():
            call_text = call_text.replace(key, random.choice(values))
        results.append(call_text)
    return results

def create_911_calls_dataset():
    # Placeholder values
    locations = ['Main St', 'Broadway', 'Elm Ave', '5th Avenue', 'Park Lane', 'Central Park', 
                 'Riverside', 'Downtown', 'Suburban area', 'Industrial zone']
    
    times = ['5 minutes ago', 'just now', '10 minutes ago', 'a few moments ago', 'recently']
    
    events = ['car accident', 'house fire', 'medical emergency', 'robbery', 'domestic disturbance', 
              'shooting', 'gas leak', 'suspicious package', 'missing child', 'flooded road']

    names = ['John', 'Emily', 'Sarah', 'Michael', 'David', 'Jessica', 'Robert', 'Laura']
    
    medical_conditions = ['heart attack', 'stroke', 'severe bleeding', 'choking', 'allergic reaction']

    # Templates for different scenarios
    critical_calls = [
        "Caller: There's been a {EVENT} at {LOCATION}. We need help immediately!\nOperator: Stay calm. Emergency services are on their way.\nCaller: Please hurry, people are hurt!",
        "Caller: My friend is having a {CONDITION} at {LOCATION}. Please send an ambulance fast!\nOperator: An ambulance is on the way. Keep your friend comfortable and stay on the line.",
        "Caller: There's a fire in my building at {LOCATION}! It's spreading quickly!\nOperator: Evacuate the building immediately. Firefighters are on their way.",
        "Caller: I heard gunshots near {LOCATION}. I think someone is hurt!\nOperator: Stay inside and stay safe. Police are on their way."
    ]

    high_priority_calls = [
        "Caller: There's a {EVENT} at {LOCATION}. It looks bad.\nOperator: Are there any injuries?\nCaller: I’m not sure, but it looks serious.",
        "Caller: We have a {CONDITION} here. It happened {TIME}. What should we do?\nOperator: Help is on the way. Please stay with the patient.",
        "Caller: I see smoke coming from a building at {LOCATION}.\nOperator: Fire services have been notified. Stay away from the area."
    ]

    medium_priority_calls = [
        "Caller: I just witnessed a minor {EVENT} on {LOCATION}. It doesn’t look like anyone is hurt.\nOperator: Thank you for the report. We'll check it out.",
        "Caller: My neighbor is acting strangely at {LOCATION}. It might be nothing, but I’m concerned.\nOperator: We'll send someone to check on the situation."
    ]

    low_priority_calls = [
        "Caller: There’s a traffic jam on {LOCATION} due to a small accident.\nOperator: Thanks for letting us know. We’ll send a traffic officer.",
        "Caller: There’s a loud party at {LOCATION}. It’s been going on for hours.\nOperator: We’ll notify local police to check it out."
    ]

    non_emergency_calls = [
        "Caller: Can you tell me if the community center is open today?\nOperator: This line is for emergencies only. Please contact the center directly.",
        "Caller: My cat is stuck in a tree at {LOCATION}. Can you help?\nOperator: Please contact animal control for assistance."
    ]

    # Placeholder dictionary
    placeholders = {
        '{LOCATION}': locations,
        '{TIME}': times,
        '{EVENT}': events,
        '{CONDITION}': medical_conditions,
        '{NAME}': names
    }

    # Data structure
    data = {
        'text': [],
        'emergency_level': [],
        'sentiment': [],
        'is_emergency': [],
        'disaster_type': [],
        'verified': [],
        'timestamp': []
    }

    # Generate calls for each priority level
    for templates, level, sentiment, is_emergency in [
        (critical_calls, 'CRITICAL', 'PANIC', True),
        (high_priority_calls, 'HIGH', 'NEGATIVE', True),
        (medium_priority_calls, 'MEDIUM', 'NEUTRAL', True),
        (low_priority_calls, 'LOW', 'NEUTRAL', True),
        (non_emergency_calls, 'NON_EMERGENCY', 'POSITIVE', False)
    ]:
        # Increase the number of variations for each template to ensure 1000+ rows
        for template in templates:
            variations = generate_911_call_variations(template, 100, placeholders)
            data['text'].extend(variations)
            data['emergency_level'].extend([level] * len(variations))
            data['sentiment'].extend([sentiment] * len(variations))
            data['is_emergency'].extend([is_emergency] * len(variations))
            
            # Assign disaster types
            if 'fire' in template.lower():
                disaster_type = 'FIRE'
            elif 'shooting' in template.lower():
                disaster_type = 'CRIME'
            elif 'medical' in template.lower():
                disaster_type = 'MEDICAL'
            elif 'accident' in template.lower():
                disaster_type = 'TRAFFIC'
            else:
                disaster_type = 'OTHER'
            
            data['disaster_type'].extend([disaster_type] * len(variations))
            data['verified'].extend(np.random.choice([True, False], size=len(variations), p=[0.8, 0.2]))
            
            # Generate timestamps
            start_date = datetime.now() - timedelta(days=30)
            timestamps = [start_date + timedelta(minutes=random.randint(0, 43200)) for _ in range(len(variations))]
            data['timestamp'].extend(timestamps)

    # Create DataFrame
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)

    # Ensure at least 1000 rows
    if len(df) < 1000:
        additional_rows = 1000 - len(df)
        df = pd.concat([df, df.sample(additional_rows, replace=True)], ignore_index=True)

    # Save dataset
    df.to_csv('911_calls_dataset_1000.csv', index=False)
    print(f"Created 911 calls dataset with {len(df)} entries.")
    return df

# Generate and save the dataset
df = create_911_calls_dataset()
print(df.head())
