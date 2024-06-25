
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from pydantic import BaseModel, Field
import ast
from pathlib import Path
from typing import Iterator, Any, Optional, List, NotRequired, Union, Tuple
import re

from langchain.prompts import load_prompt
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from pathlib import Path
from langchain.memory import VectorStoreRetrieverMemory
from typing import List, Tuple
from langchain.schema import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, MarkdownTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import format_document, PromptTemplate, ChatMessagePromptTemplate, ChatPromptTemplate

import dotenv
dotenv.load_dotenv()


def unwrap_docs(text: str) -> str:
	test = text.strip()
	if test.startswith('"""') and test.endswith('"""'):
		test = test[3:-3].strip()
		return test
	return text


def extract_md_blocks(text: str) -> List[str]:
	pattern = r"```(?:\w+\s+)?(.*?)```"
	matches = re.findall(pattern, text, re.DOTALL)
	res = [unwrap_docs(block.strip()) for block in matches]
	if not res:
		test = unwrap_docs(text)
		if test:
			return [test]

	return res


llm = ChatOpenAI(
	model_name="gpt-4o",  # type: ignore
	openai_api_key=os.environ["OPENAI_KEY"],  # type: ignore
	openai_organization="org-Mxa66lw5lFVUrdKlbb4gJYsv",  # type: ignore
	temperature=0.3,
	streaming=True,
	max_retries=10,
	verbose=True)


class GroupDir(BaseModel):
	files: List[str]
	path: str


def list_all_files(path: Path) -> Iterator[str]:
	ext = ['.py', '.md', '.rst']
	subfolders: List[str] = []

	path = path.absolute()

	for f in os.scandir(path):
		if f.is_dir():
			subfolders.append(f.path)
		if f.is_file():
			if os.path.splitext(f.name)[1].lower() in ext:
				yield f.path

	for dir in subfolders:
		new_path = Path.joinpath(path, dir)

		for r in list_all_files(new_path):
			yield r


embedding_model = OpenAIEmbeddings(model="text-embedding-3-large",
								api_key=os.environ["OPENAI_KEY"],  # type: ignore
								organization="org-Mxa66lw5lFVUrdKlbb4gJYsv")

# # The vectorstore to use to index the child chunks
# child_vectorstore = Chroma(
#     collection_name="parent_document_splits",
#     embedding_function=embedding_model
# )

# parent_docstore = InMemoryStore()

# retriever_parent = ParentDocumentRetriever(
#     vectorstore=child_vectorstore,
#     docstore=parent_docstore,
#     child_splitter=child_splitter,
#     parent_splitter=parent_splitter)


def chunk_python(base_path: Path, file: str) -> Tuple[Document, List[Document]]:

	file = str(base_path.joinpath(file).absolute())
	file_id = str(Path(file).relative_to(base_path))

	file_content: str
	with open(file, 'r', encoding='utf-8') as f:
		file_content = f.read()

	file_content_lines = re.split(r'\r?\n', file_content)
	file_content = '\n'.join(file_content_lines)

	main_doc = Document(page_content=file_content,  metadata={"id": file_id})
	sub_docs: List[Document] = []

	ast_result = ast.parse(file_content)
	visitor = MyNodeVisitor(filecontent=file_content, lines=file_content_lines)
	visitor.visit(ast_result)

	def _pick(elem: Elem | None) -> Iterator[Elem]:
		if not elem:
			return

		if isinstance(elem, Container):
			for subelem in reversed(elem.children):
				for res in _pick(subelem):
					yield res

			if isinstance(elem, ClassElem) and elem.constructor:
				for res in _pick(elem.constructor):
					return res

		yield elem

	for to_document in _pick(visitor._module):
		print("picked ", to_document.get_id())
		doc = Document(page_content=to_document.body, metadata={**main_doc.metadata, "id": f'{file_id}#{to_document.get_id()}', "doc_id": file_id})
		sub_docs.append(doc)

	return (main_doc, sub_docs)


def markdown_chunking(base_path: Path, file: str) -> Tuple[Document, List[Document]]:
	file = str(base_path.joinpath(file).absolute())
	file_id = str(Path(file).relative_to(base_path))

	headers_to_split_on = [
		("#", "Header 1"),
		("##", "Header 2"),
	]
	markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

	document_content = get_file_content(base_path.joinpath(file).absolute())
	document_metadata = {"id": file_id}
	document = Document(page_content=document_content, metadata=document_metadata)
	md_header_splits = markdown_splitter.split_text(document_content)
	for idx, doc in enumerate(md_header_splits):
		doc.metadata = {**document_metadata, "id": f'{file_id}#{idx}', 'doc_id': file_id}

	return (document, md_header_splits)


def rst_chunking(base_path: Path, file: str) -> Tuple[Document, List[Document]]:
	file = str(base_path.joinpath(file).absolute())
	file_id = str(Path(file).relative_to(base_path))

	document_content = get_file_content(base_path.joinpath(file).absolute())
	document_metadata = {"id": file_id}
	document = Document(page_content=document_content, metadata=document_metadata)

	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=1000,
		chunk_overlap=20,
		length_function=len,
		separators=[
			"\n\n",
			"\n",
			" ",
			".",
			",",
			"\u200b",  # Zero-width space
			"\uff0c",  # Fullwidth comma
			"\u3001",  # Ideographic comma
			"\uff0e",  # Fullwidth full stop
			"\u3002",  # Ideographic full stop
			"",
		],
	)

	rst_splits = text_splitter.split_documents([document])
	for idx, doc in enumerate(rst_splits):
		doc.metadata = {**document_metadata, "id": f'{file_id}#{idx}', "doc_id": file_id}

	return (document, rst_splits)

from langchain.retrievers import ParentDocumentRetriever
def main(base_path: Path):

	main_doc_store = InMemoryStore()
	chunks_store = DocArrayInMemorySearch.from_params(embedding=embedding_model)
	base_path = base_path.absolute()
	for idx, file in enumerate(list_all_files(base_path)):
		# if idx > 10:
		# 	break
		print(file)

		r, ext = os.path.splitext(file)
		if ext == '.py':
			try:
				main_doc, chunks = chunk_python(base_path=base_path, file=file)
			except:
				print("errror while opening ", file)
				main_doc, chunks = rst_chunking(base_path=base_path, file=file)
				
		elif ext == '.md':
			main_doc, chunks = markdown_chunking(base_path=base_path, file=file)
		else:
			main_doc, chunks = rst_chunking(base_path=base_path, file=file)

		main_doc_store.mset([(main_doc.metadata["id"], main_doc)])
		chunks_store.add_documents(chunks)
		
		
	retriever = ParentDocumentRetriever(
		vectorstore=chunks_store,
		docstore=main_doc_store,
		child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
	)
	
	doc_prompt = PromptTemplate.from_template(template="## File '{id}'\n```\n{page_content}\n```")
	def format(docs: List[Document]) -> str:
		return '\n\n'.join(format_document(doc, doc_prompt) for doc in docs)
	
	
	main_prompt =  ChatPromptTemplate.from_messages([
		("system", "You are a helpful assistant answering questions about a project coded in python, which contains readme and rst files as well. You must answser only from the provided context, if you can't say so to the user. Do not answser questions non related to the project."),
		("system", "Always add the file name associated to the source you used to answser the question"), 
		("user", "# Context\nHere are some files that may help you answering the question of the user.\n{context}\n\n"),
		("user", "Question: {input}")
	])
	
	chain = {
		"context": retriever | format,
		"input": RunnablePassthrough()
	} | main_prompt | llm | StrOutputParser()
	
	import sys
	while True:
		print("Ask you stupid question")
		line = sys.stdin.readline().strip()
		if not line:
			continue
		
		print("Answer from LLM: ", chain.invoke(line))
		
		
	


def get_file_content(file_path):
	with open(file_path, 'r', encoding='utf-8') as file:
		return file.read()


class Elem(BaseModel):
	name: str
	doc: Optional[str]
	parent: Optional['Container']
	node_line: int
	node_col: int
	body: str

	def get_id(self) -> str:
		return self.parent.get_id() + '.' + self.name if self.parent else self.name

	@staticmethod
	def get_node(container: 'Elem', id: str) -> Optional['Elem']:
		parts = id.split('.')
		if len(parts) < 2:
			if container.name == id:
				return container
			return None

		if container.name != parts[0]:
			return None

		if isinstance(container, Container):
			for child in container.children:
				res = Elem.get_node(child, '.'.join(parts[1:]))
				if res:
					return res

		return None


class Container(Elem):
	children: List['Elem'] = Field(default_factory=lambda: [])


class ModuleContainer(Container):
	def get_id(self) -> str:
		return ""


class ClassElem(Container):
	constructor: Optional['FunctionElem']
	functions: List['FunctionElem'] = Field(default_factory=lambda: [])


class FunctionElem(Container):
	node_line: int
	node_col: int


class MyNodeVisitor(ast.NodeTransformer):

	def __init__(self, filecontent: str, lines: List[str]):
		self._filecontent = filecontent
		self._module: Optional[ModuleContainer] = None
		self._current_node = self._module
		self._lines = lines

	def visit_Module(self, node: ast.Module):
		if self._module:
			raise Exception()

		self._current_node = self._module = ModuleContainer(name='module', doc=ast.get_docstring(node), parent=None, node_col=1, node_line=1, body=self._filecontent)
		return ast.NodeVisitor.generic_visit(self, node)

	def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
		if not self._current_node:
			raise Exception()

		parent = self._current_node
		fct = FunctionElem(
			name=node.name,
			body=ast.unparse(node),
			doc=ast.get_docstring(node),
			parent=self._current_node,
			node_line=node.body[0].lineno,
			node_col=node.body[0].col_offset)

		if isinstance(self._current_node, ClassElem):
			if fct.name == '__init__':
				self._current_node.constructor = fct
			else:
				self._current_node.functions.append(fct)
				self._current_node.children.append(fct)
		else:
			self._current_node.children.append(fct)

		self._current_node = fct
		res = ast.NodeVisitor.generic_visit(self, node)
		self._current_node = parent
		return res

	def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
		if not self._current_node:
			raise Exception()

		parent = self._current_node
		fct = FunctionElem(
			name=node.name,
			body=ast.unparse(node),
			doc=ast.get_docstring(node),
			parent=self._current_node,
			node_line=node.body[0].lineno,
			node_col=node.body[0].col_offset)

		if isinstance(self._current_node, ClassElem):
			if fct.name == '__init__':
				self._current_node.constructor = fct
			else:
				self._current_node.functions.append(fct)
				self._current_node.children.append(fct)
		else:
			self._current_node.children.append(fct)

		self._current_node = fct
		res = ast.NodeVisitor.generic_visit(self, node)
		self._current_node = parent
		return res

	def visit_ClassDef(self, node: ast.ClassDef) -> Any:
		if not self._current_node:
			raise Exception()

		parent = self._current_node
		class_elem = ClassElem(name=node.name, body=ast.unparse(node), parent=parent, doc=None,
							constructor=None, node_line=node.body[0].lineno, node_col=node.body[0].col_offset)
		self._current_node = class_elem
		parent.children.append(self._current_node)
		res = ast.NodeVisitor.generic_visit(self, node)
		self._current_node = parent

		class_elem.doc = ast.get_docstring(node)
		return res


doc, chunks = chunk_python(Path('./resources/cookiecutter/').absolute(), 'cookiecutter/cli.py')
# print(chunks[0].metadata)

main(Path('./resources/cookiecutter/'))
