import os
import atlassian
from atlassian import Confluence
import logging

# Configure logging (you can customize the level and format as needed)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_url():
    """
    Retrieves the Confluence URL from environment variables.

    Returns:
        str or None: The Confluence URL or None if not found.
    """

    url = os.getenv("confluence-url")
    if not url:
        logging.error("Error: 'confluence-url' environment variable not found.")
        return None
    return url

def get_spaces(confluence):
    """
    Retrieves a list of spaces from Confluence.

    Args:
        confluence: A Confluence API client instance.

    Returns:
        list: A list of dictionaries representing the spaces, 
              each containing 'key' and 'name' attributes.
    """

    try:
        all_spaces = []
        start = 0
        limit = 10  # Adjust as needed
        while True:
            results = confluence.get_all_spaces(start=start, limit=limit)
            all_spaces.extend(results['results'])

            if '_links' in results and 'next' in results['_links']:
                start += limit
            else:
                break

        spaces = [{'key': space['key'], 'name': space['name']} for space in all_spaces]
        return spaces

    except atlassian.errors.ApiError as e:
        logging.error(f"Error fetching spaces from Confluence: {e}")
        return []

def connect_to_Confluence():
    """
    Connect to Confluence using Basic Authentication (email and API token).

    Returns:
        Confluence or None: A connector to Confluence or None if the connection fails.
    """

    try:
        url = get_url() 
        if url is None:
            return None

        username = os.getenv("confluence-username")
        api_token = os.getenv("confluence-api-token")

        confluence = Confluence(
            url=url,
            username=username,
            password=api_token,
            cloud=True
        )

        return confluence

    except atlassian.errors.ApiPermissionError as e:
        logging.error(f"Error: Insufficient permissions to access Confluence: {e}")
        return None

    except atlassian.errors.ApiError as e:
        logging.error(f"Error connecting to Confluence: {e}")
        return None

def get_all_pages(confluence, space):
    '''
    Get all the pages within the specified Confluence space.

    Args:
        confluence: A Confluence API client instance.
        space: The key of the Confluence space.

    Returns:
        list: A list of page objects. Each page object contains information about a Confluence page.
    '''

    keep_going = True
    start = 0
    limit = 100
    pages = []
    while keep_going:
        results = confluence.get_all_pages_from_space(space, start=start, limit=limit, status=None, expand='body.storage,history', content_type='page')

        for page in results:
            try:
                if 'createdBy' in page['history']:
                    page['creator'] = page['history']['createdBy']['displayName']
                else:
                    page['creator'] = "Unknown"

                if 'createdDate' in page['history']:
                    page['created_date'] = page['history']['createdDate']
                else:
                    page['created_date'] = "Unknown"

                if 'lastUpdated' in page['history']:
                    page['last_updated'] = page['history']['lastUpdated']['when']
                else:
                    page['last_updated'] = "Unknown"

                if 'numberContentViews' in page['history']:
                    page['views'] = page['history']['numberContentViews']
                else:
                    page['views'] = 0

            except atlassian.errors.ApiError as e:
                logging.error(f"Error fetching details for page '{page['title']}': {e}")
                page['creator'] = "Unknown"
                page['created_date'] = "Unknown"
                page['last_updated'] = "Unknown"
                page['views'] = -1 

        pages.extend(results)
        if len(results) < limit:
            keep_going = False
        else:
            start = start + limit
    return pages