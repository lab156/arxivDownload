import sqlalchemy as sa
from lxml import etree
import subprocess
import process

dataset_loc = '/mnt/dataset-arXMLiv-08-2018'

def locate_wait_split(searchable):
    run_res = subprocess.Popen(['locate', searchable, '-b'], stdout=subprocess.PIPE)
    run_res.wait()
    if not run_res.returncode:
        stdout_bytes = run_res.communicate()[0]
        stdout_lst = stdout_bytes.decode('utf8').split('\n')
        for f in stdout_lst:
            if dataset_loc in f:
                return stdout_lst[0]
        else:
            return None
    else:
        return None

if __name__ == "__main__":
    import sys
    xml = etree.parse(sys.argv[1])
    for k,art in enumerate(xml.iter('article')):
        # tar2api example input: 'http://arxiv.org/abs/1801.00137v1' to 1801.00137
        if not art.attrib.get('searched'):
            article_url = art.find('id').text
            print('Searching for %s                          '%article_url, end='\r')
            searchable = process.Tar2api(article_url, sep='')
            loc = locate_wait_split(searchable)
            # Add the location to the article tag
            if loc:
                location = etree.SubElement(art, 'location')
                location.text = loc
                art.attrib['searched'] = "True"
            else:
                art.attrib['searched'] = "False"
        if k%500 == 0:
            with open(sys.argv[1], 'w+') as xml_file:
                print(etree.tostring(xml, pretty_print=True).decode('utf8'),file=xml_file)
    #Also save when forloop is over
    with open(sys.argv[1], 'w+') as xml_file:
        print(etree.tostring(xml, pretty_print=True).decode('utf8'),file=xml_file)

