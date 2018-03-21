"""

This is a Unit Testing file for model_development.py. 

Author: Yiwei Sun

"""

def test_binary_creater():
    """Test binary_breater function."""
    
    # Expected output
    expected_df = pd.DataFrame(data={'is_toxic': [1.0,1.0]})
    
    try:
        # Check function output
        assert (binary_creater()['is_toxic'].head(n=2).equals(\
                expected_df['is_toxic']))
        print('binary_creater() function test passed!')
        
    except:
        print('binary_creater() function test failed!')
    
