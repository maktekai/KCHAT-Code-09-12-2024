from fastapi import  Query
import sqlite3
import ast
import math
import json
import random
import secrets
import smtplib
import ssl
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import stripe
import tiktoken as tiktoken
from fastapi import FastAPI, HTTPException, status,Request
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, desc,DateTime, Boolean, Float,JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from sqlalchemy.sql import func
from typing import Optional, Tuple
from datetime import datetime
from typing import List,Dict
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from passlib.context import CryptContext
from fastapi.responses import RedirectResponse

load_dotenv()  # take environment variables from .env.
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

from CreateKnowledgeStore import create_chatbot, GetDocumentsFromURL, TestAPISQouta
from DeleteKnowledgeStore import deleteVectorsusingKnowledgeBaseID
from ChatChain import Get_Conversation_chain

# Create SQLite database engine
SQLALCHEMY_DATABASE_URL = "sqlite:///./Database.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for SQLAlchemy models
Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Define SQLAlchemy model for the entry

class Users(Base):
    __tablename__ = "users"
    user_id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    name= Column(String)
    email = Column(String, unique=True, index=True)
    password=Column(String)
    isVerified=Column(Boolean,default=False)
    verification_token = Column(String, unique=True, nullable=True)
    redirected=Column(Boolean,default=False)
    isAdmin=Column(Boolean,default=False)
    registerd_at= Column(DateTime, default=func.now())

class Selected_Plan(Base):
    __tablename__ = "Selected_Plan"
    Selected_Plan_ID = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    plan_id= Column(String)
    user_id= Column(String)
    subscription_id=Column(String)
    customer_id=Column(String)
    purchased_at = Column(DateTime, default=datetime.utcnow)
    last_updated= Column(DateTime, default=datetime.utcnow)
    message=Column(String,default="Subscribed")
    status=Column(Boolean, default=True)

class Consumption(Base):
    __tablename__ = "Consumption"
    consumption_id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    plan_id=Column(Integer)
    user_id=Column(Integer)
    consumed_chatbots=Column(Integer,default=0)
    consumed_stores=Column(Integer,default=0)
    consumed_store_tokens=Column(Integer,default=0)
    consumed_chatbot_response_tokens=Column(Integer,default=0)
    last_updated= Column(DateTime, default=func.now())

class Plans(Base):
    __tablename__ = "Plans"
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    plan_id=Column(String)
    plan_Name = Column(String)
    total_chatbots_allowed=Column(Integer)
    total_knowldegeStores_allowed=Column(Integer)
    total_knowldegStores_Allowed_Tokens=Column(Integer)
    Total_Responce_Tokens_allowed=Column(Integer)
    price=Column(Integer)
    added_at = Column(DateTime, default=func.now())

class knowledge_Store(Base):
    __tablename__ = "knowledge_Stores"
    knowledge_base_id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer)
    descriptive_name= Column(String)
    xml_url = Column(String)
    wordpress_base_url = Column(String)
    syncing_feature = Column(Integer, default=0)
    syncing_period = Column(Integer, default=0)
    syncing_state = Column(Integer, default=0)


class ChatBotsConfigurations(Base):
    __tablename__ = "chatBots_configurations"
    chatbot_id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer)
    descriptive_name= Column(String)
    temperature = Column(String)
    llm = Column(String)
    knowledgeBases=Column(String)
    OutBoundLinks=Column(String)
    AddContext=Column(Integer)


class ChatbotAppearnace(Base):
    __tablename__ = "ChatbotAppearnace"
    chatbot_appeance_id =Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    chatbot_id = Column(Integer)
    ThemeColor=Column(String)
    InitialMessage=Column(String)
    DisplayName=Column(String)

class syncingJob(Base):
    __tablename__ = "Sycing_Jobs"
    job_id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    knowledge_base_id = Column(Integer)
    user_id = Column(Integer)
    xml_url=Column(String)
    wordpress_base_url = Column(String)
    syncing_period = Column(Integer, default=0)
    last_performed = Column(DateTime, default=func.now())

class ChatLogs(Base):
    __tablename__ = "ChatLogs"
    Message_id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    chatbot_id = Column(Integer)
    visitor_ID = Column(Integer)
    Human_Message = Column(String)
    AI_Responce = Column(String)
    context = Column(String)
    responded_at = Column(DateTime, default=datetime.utcnow)

class LeadsGenerated(Base):
    __tablename__ = "LeadsGenerated"
    generated_leads_id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    chatbot_id = Column(Integer)
    name  = Column(String)
    email = Column(String)
    phone = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class PaymentsTransactions(Base):
    __tablename__ = "PaymentsTransactions"
    payment_id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id=Column(String)
    plan_id=Column(String)
    transactionInfo=Column(JSON)
    tarnsactiondate = Column(DateTime, default=datetime.utcnow)

# Create tables in the database
Base.metadata.create_all(bind=engine)

class user(BaseModel):
    name:str
    email:str
    password:str
    redirected:bool

# Define Pydantic model for request body
class knowledgeStoreCreate(BaseModel):
    user_id: str
    descriptive_name:str
    xml_url: str
    wordpress_base_url: str
    syncing_feature: int
    syncing_period: int
    syncing_state: int
    
class knowledgeStoreEdit(BaseModel):
    user_id: str
    knowledge_base_id:str
    descriptive_name:str
    xml_url: str
    wordpress_base_url: str
    syncing_feature: int
    syncing_period: int
    syncing_state: int
    
class syncingFeatureStatus(BaseModel):
    knowledgeStoreId: str
    syncPeriod: int

class ChatBots(BaseModel):
    user_id :str
    descriptive_name:str
    temperature:str
    llm:str
    knowledgeBases: str
    OutBoundLinks:str
    AddContext:int

class EditChatBots(BaseModel):
    descriptive_name:str
    temperature:str
    llm:str
    knowledgeBases: str
    OutBoundLinks:str
    AddContext:int

class EditAppeanceChatBots(BaseModel):
    ThemeColor:str
    InitialMessage:str
    DisplayName:str

class AddLeadsPydanticModel(BaseModel):
    chatbot_id:str
    name: Optional[str]=""
    email: Optional[str]=""
    phone: Optional[str]=""

class PlansPydnaticModel(BaseModel):
    plan_id:str
    plan_Name:str
    total_chatbots_allowed:int
    total_knowldegeStores_allowed:int
    total_knowldegStores_Allowed_Tokens:int
    Total_Responce_Tokens_allowed:int
    price:int

class GetPost(BaseModel):
    rawText:str

class ChatRequest(BaseModel):
    chatbotId: str
    question: str
    visitorID:int
    chat_history: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    answer: str
    reference_context: List[dict]


class CheckPlanExistance(BaseModel):
    plan_id: str
    user_id: str

class passwordreset(BaseModel):
    token: str
    new_password: str

class query_obj(BaseModel):
    sql_query: str

class CreateCheckoutSession(BaseModel):
    user_id: str
    plan_id: str  # This is the Stripe Price ID for the plan


class UserUpdate(BaseModel):
    name: str
    email: str
    password: Optional[str]=""
    isVerified: bool
    verification_token: str
    redirected: bool

class UserResponse(BaseModel):
    user_id: str
    name: str
    email: str
    isVerified: bool
    verification_token: str
    redirected: bool
    registerd_at: datetime

    class Config:
        orm_mode = True


class UserInfo(BaseModel):
    name: str
    email: str
    password: Optional[str]=""
    isVerified: bool
   
class PaginatedResponse(BaseModel):
    data: List[UserResponse]
    status: str
    totalPages: int
    currentPage: int

# Stripe API key setup
stripe.api_key = os.getenv('STRIPE_KEY')
app=FastAPI()




# Initialize FastAPI app
if os.getenv("ENV") == "production":
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
else:
    app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
BASE_URL="https://api.optimalaccess.com/"
#BASE_URL="http://127.0.0.1:8000/"

@app.get("/users_by_pagination/", response_model=PaginatedResponse)
def read_users_by_pagination(
    pageNo: int = Query(1, alias="pageNo"), 
    records: int = Query(10, alias="records")
):
    db = SessionLocal()
    skip = (pageNo - 1) * records
    users = db.query(Users).offset(skip).limit(records).all()
    count = db.query(Users).count()

    total_pages = math.ceil(count / records)
    return {
        "data": users,
        "status": "ok",
        "totalPages": total_pages,
        "currentPage": pageNo,
    }
@app.get("read_users/", response_model=List[UserResponse])
def read_users(skip: int = 0, limit: int = 10):
    db = SessionLocal()
    users = db.query(Users).offset(skip).limit(limit).all()
    return users
@app.delete("/deleteusers/{user_id}")
def delete_user(user_id: str):
    db = SessionLocal()
    db_user = db.query(Users).filter(Users.user_id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    selected_plans = db.query(Selected_Plan).filter(Selected_Plan.user_id == user_id).all()
    for plan in selected_plans:
         deactiavtePlan(plan.subscription_id)
    
    db.query(Selected_Plan).filter(Selected_Plan.user_id == user_id).delete()
    db.query(Consumption).filter(Consumption.user_id == user_id).delete()
    db.query(knowledge_Store).filter(knowledge_Store.user_id == user_id).delete()
    db.query(ChatBotsConfigurations).filter(ChatBotsConfigurations.user_id == user_id).delete()
    db.query(ChatbotAppearnace).filter(ChatbotAppearnace.chatbot_id == user_id).delete()
    db.query(syncingJob).filter(syncingJob.user_id == user_id).delete()
    db.query(ChatLogs).filter(ChatLogs.visitor_ID == user_id).delete()
    db.query(LeadsGenerated).filter(LeadsGenerated.chatbot_id == user_id).delete()
    db.query(PaymentsTransactions).filter(PaymentsTransactions.user_id == user_id).delete()

    db.delete(db_user)
    db.commit()
    return {"detail": "User deleted"}

@app.get("/read_single_users/{user_id}", response_model=UserResponse)
def read_user(user_id: str):
    db = SessionLocal()
    db_user = db.query(Users).filter(Users.user_id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@app.put("/update_user/{user_id}", response_model=UserResponse)
def update_user(user_id: str, request: UserInfo):
    db = SessionLocal()
    try:
        user = db.query(Users).filter(Users.user_id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if request.email:
            existing_user = db.query(Users).filter(Users.email == request.email).first()
            if existing_user and existing_user.user_id != user_id:
                raise HTTPException(status_code=400, detail="Email already registered. Please use a different email address.")

        update_data = request.dict(exclude_unset=True)
        if 'password' in update_data:
            update_data['password'] = pwd_context.hash(update_data['password'])

        for key, value in update_data.items():
            setattr(user, key, value)

        db.commit()
        db.refresh(user)

        return user

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.post("/create_user/")
def create_user(request: UserInfo):
    try:
        db = SessionLocal()
        existing_user = db.query(Users).filter(Users.email == request.email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered Please use any other Email Address or Login with Existing Email Address.")
        request.password = pwd_context.hash(request.password)
        db_entry = Users(**request.dict())
        db.add(db_entry)
        db.commit()
        db.refresh(db_entry)
        db_entry.verification_token=generate_verification_token()
        db.commit()

        if request.isVerified:
            db_entry.isVerified = True
            db.commit()
            db.close()
            return {"status": "ok",
                    "message": "User registered successfully.",                    
                    "data": db_entry}
        else:
            send_verifiaction_code_on_email(db_entry.email,db_entry.name,db_entry.verification_token)
            db.commit()
            db.close()
            return {"status": "ok",
                    "message": "User registered successfully. Check your email for verification instructions.",
                    "data": db_entry}

    except Exception as e:
            db.close()
            return {"status": "error","message": str(e), "data": None}

@app.post("/Checkout_Completed")
async def Checkout_Completed(request: Request):
  payload = await request.body()
  sig_header = request.headers.get('Stripe-Signature')
  endpoint_secret = "whsec_wd5tII1R1pp5YxbK6mnY6zpa7p2umtiV"

  try:
    event = stripe.Webhook.construct_event(payload, sig_header,
                                           endpoint_secret)
  except ValueError as e:
    raise HTTPException(status_code=400, detail="Invalid payload")
  except stripe.error.SignatureVerificationError as e:
    raise HTTPException(status_code=400, detail="Invalid signature")
  # Handle the event
  if event['type'] == 'checkout.session.completed':
    session = event['data']['object']
    customer_id = session['customer']
    user_id = session['metadata']['user_id']

    # Store the subscription data in your database
    subscription_id = session['subscription']
    subscription = stripe.Subscription.retrieve(subscription_id)
    plan_id = subscription['items']['data'][0]['price']['id']
#    print(subscription)
    result= subscribePlan(user_id, customer_id, subscription_id, plan_id)
  return result

@app.post("/SubscriptionCancled")
async def SubscriptionCancled(request: Request):
  payload = await request.body()
  sig_header = request.headers.get('Stripe-Signature')
  endpoint_secret = "whsec_9UzxvAb4csE0qR9j7TXRWL2gqdrkcB7O"

  try:
    event = stripe.Webhook.construct_event(payload, sig_header,
                                           endpoint_secret)
  except ValueError as e:
    raise HTTPException(status_code=400, detail="Invalid payload")
  except stripe.error.SignatureVerificationError as e:
    raise HTTPException(status_code=400, detail="Invalid signature")
  if event['type'] == 'customer.subscription.deleted':
        subscription = event['data']['object']['id']
        deactiavtePlanAfterPeriod(subscription)
  return True


@app.post("/create-checkout-session")
async def create_checkout_session(data: CreateCheckoutSession):
    try:
        db = SessionLocal()
        planExists = db.query(Plans).filter(Plans.plan_id == data.plan_id).first()
 #       print(planExists)
        if planExists:
            existing_plan = db.query(Selected_Plan).filter(Selected_Plan.plan_id == data.plan_id,
                                                           Selected_Plan.user_id == data.user_id).first()
            if not  existing_plan or  existing_plan.status==False:
                email=getEmailInfoFromSelectedPlans(data.user_id)
                session = stripe.checkout.Session.create(
                    payment_method_types=['card'],
                    line_items=[{
                        'price': data.plan_id,
                        'quantity': 1,
                    }],
                    mode='subscription',
                    success_url='https://kchat.website?session_id={CHECKOUT_SESSION_ID}',
                    cancel_url='https://kchat.website?status=failed',
                    metadata={
                        'user_id': data.user_id  # Store the user ID in the metadata
                    },
                     customer_email=email  # Include the customer's email address

                )
 #               print(session.id)
                return {"sessionId": session.id}

            else:
                
                raise HTTPException(status_code=400, detail="Provided Plan Exists for the user You cannot Stack Same Plans Try another Plan.")

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute_query/")
def execute_query(query: query_obj):
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect("Database.db")
        cursor = conn.cursor()
        
        # Execute the provided SQL query
        cursor.execute(query.sql_query)
        
        # Fetch all results
        results = cursor.fetchall()
        
        # Get the column names
        column_names = [description[0] for description in cursor.description]
        
        # Close the connection
        conn.close()
        
        # Return results as a list of dictionaries
        return [dict(zip(column_names, row)) for row in results]
    except sqlite3.Error as e:
        raise HTTPException(status_code=400, detail=str(e))

def subscribePlan(user_id,customer_id,new_subscription_id,plan_id):
    try:
        db = SessionLocal()
        planExists = db.query(Plans).filter(Plans.plan_id == plan_id).first()
        if planExists:
  #          print(user_id,customer_id)
            other_plans = db.query(Selected_Plan).filter(Selected_Plan.user_id ==user_id).all()
            for planObj in list(other_plans):
                if planObj.status:
                   deactiavtePlan(planObj.subscription_id)
   #                print("after deactivte")
            existing_plan = db.query(Selected_Plan).filter(Selected_Plan.plan_id == plan_id,
                                                       Selected_Plan.user_id == user_id).first()

            if not existing_plan:
                #print("after exisitn plan")
                #other_plans = db.query(Selected_Plan).filter(Selected_Plan.user_id ==user_id).all()
                #for planObj in list(other_plans):
                #    deactiavtePlan(planObj.subscription_id)
                #    print("after deactivte")
                plan_entry = Selected_Plan(user_id=user_id, plan_id=plan_id,customer_id=customer_id,subscription_id=new_subscription_id)
                db.add(plan_entry)
    #            print(plan_entry)
                db.commit()

                plan_consumpiton_entry = Consumption(user_id=user_id, plan_id=plan_id)
                db.add(plan_consumpiton_entry)
                db.commit()
                db.close()
     #           print("doneeeeeeeeeeeeeeeeeeeeeeeeeeeee")
                return {"status": "ok", "message": "Plan Added to the Your Plans Pool successfully and Previous Plans got Deactivated.", "data": None}
            else:
                #other_plans = db.query(Selected_Plan).filter(Selected_Plan.user_id ==user_id).all()
                #for planObj in list(other_plans):
                #    deactiavtePlan(planObj.subscription_id)
                #    print("after deactivte")
                entry = db.query(Selected_Plan).filter(Selected_Plan.user_id ==user_id,Selected_Plan.plan_id==plan_id).first()
                
                if not entry.status:
                   entry.status=True
                   entry.message="Renewed"
                   entry.customer_id=customer_id
                   entry.subscription_id=new_subscription_id
                   entry.last_updated=datetime.utcnow()
                   db.add(entry)
                   db.commit()
                   db.close()
 
               
    except Exception as e:
            print(e)
            db.close()
            return {"status": "error","message": str(e), "data": None}


def getInfoFromSelectedPlans(email):
    db = SessionLocal()
    existing_plan = db.query(Users).filter(Users.email == email).first()
    return existing_plan.user_id

def getEmailInfoFromSelectedPlans(user_id):
    db = SessionLocal()
    existing_plan = db.query(Users).filter(Users.user_id == user_id).first()
    return existing_plan.email

@app.post("/Completed_Payment")
async def Completed_Payment(request: Request):
    import time
    time.sleep(5)
    payload = await request.body()
    sig_header = request.headers.get('Stripe-Signature')
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, "whsec_S0QIYyBdCXwyPXGmtHLLVdfZ3IQPzLXz"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        raise HTTPException(status_code=400, detail="Invalid signature")
    db = SessionLocal()
    # Handle the event
    if event['type'] == 'invoice.payment_succeeded':
        print("tiggred ")
        invoice = event['data']['object']
        customer_id = invoice['customer']
        subscription_id = invoice['subscription']
        plan_id=get_price_id_from_invoice(invoice)
      #  print(invoice)
       # print(subscription_id,customer_id)
        #print("\n\n----------------------------------------\n\n",invoice,"\n\n------------------------")
        user_id=getInfoFromSelectedPlans(invoice['customer_email'])
        transactionEntry = PaymentsTransactions(user_id=user_id, plan_id=plan_id,transactionInfo=invoice)
        db.add(transactionEntry)
        db.commit()
        print("added once")
        updatetheConsumptionAfterPlanInACtivation(db,plan_id,user_id)
        db.close()
    # Return a success response
    return {"status": "success"}

def get_price_id_from_invoice(invoice):
    """
    Extracts the price_id from a given invoice object.

    :param invoice: The invoice object from which to extract the price_id.
    :return: The price_id if found, otherwise None.
    """
    line_items = invoice.get('lines', {}).get('data', [])
    for item in line_items:
        price_id = item.get('price', {}).get('id')
        if price_id:
            return price_id
    return None

@app.post("/checkifPlanExists/")
async def checkifPlanExists(request: CheckPlanExistance):
    try:
        db = SessionLocal()
        planExists = db.query(Plans).filter(Plans.plan_id == request.plan_id).first()
        if planExists:
            existing_plan = db.query(Selected_Plan).filter(Selected_Plan.plan_id == request.plan_id,
                                                                  Selected_Plan.user_id == request.user_id).first()
            if not existing_plan:
                db.close()
                return {"status": "ok", "message": "You can add this Plan it doesnt exist in plan pool.", "data": {"PlanStack":True}}
            else:
                return {"status": "ok", "message": "Plan Already Existed cannot stack Multiple Plans Try another Plan.", "data": {"PlanStack":False}}
        else:
            raise HTTPException(status_code=400, detail="Provided Plan Do not Exists.")

    except Exception as e:
            db.close()
            return {"status": "error","message": str(e), "data": None}

@app.get("/GetAllPlans/")
async def GetAllPlans():
    try:
        db = SessionLocal()
        entries = db.query(Plans).all()
        if entries:
            db.close()
            return {"status": "ok", "message": "Our Plans Data returned Successfully.", "data": entries}
        else:
            raise HTTPException(status_code=404, detail="Our Plans not found.")
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}
def getAggreatedPlansConsumptionByUsersPlanList(db,user_id):
    entries = db.query(Selected_Plan).filter(Selected_Plan.user_id == user_id).all()
    if not entries:
        raise HTTPException(status_code=404, detail="No Plans Found under Given user ID.")
    accamulated_chatbots = 0
    accamulated_responce_tokens_credits = 0
    accamulated_knowledgeStore_Tokens = 0
    accamulated_no_of_knowledge_Store = 0
    accamulated_price_paid = 0
    total_active_plans = 0
    for entry in entries:
        planInfo = db.query(Plans).filter(Plans.plan_id == entry.plan_id).first()
        if entry.status:
            total_active_plans += 1
            accamulated_chatbots += planInfo.total_chatbots_allowed
            accamulated_responce_tokens_credits += planInfo.Total_Responce_Tokens_allowed
            accamulated_knowledgeStore_Tokens += planInfo.total_knowldegStores_Allowed_Tokens
            accamulated_no_of_knowledge_Store += planInfo.total_knowldegeStores_allowed
            accamulated_price_paid += planInfo.price

    aggregatedPlan={
                        "user_id": user_id,
                        "Total_Plans": len(entries),
                        "Total_active_Plans": total_active_plans,
                        "Total_inactive_Plans": len(entries) - total_active_plans,
                        "Aggregated_Chatbots":accamulated_chatbots,
                        "Aggregated_Monthly_Chatbot_Response_Tokens":accamulated_responce_tokens_credits,
                        "Aggregated_Knowledge_Store_Tokens":accamulated_knowledgeStore_Tokens ,
                        "Aggregated_Knowledge_Store":accamulated_no_of_knowledge_Store,
                        "Aggregated_Cost":accamulated_price_paid,

                    }
    aggreagate_consumption=db.query(Consumption).filter(Consumption.user_id == user_id).first()
    return  aggregatedPlan,aggreagate_consumption

def getAggreatedPlansInfoByUsersPlanList(db,user_id,entries):
    accamulated_chatbots = 0
    accamulated_responce_tokens_credits = 0
    accamulated_knowledgeStore_Tokens = 0
    accamulated_no_of_knowledge_Store = 0
    accamulated_price_paid = 0
    total_active_plans=0
    for entry in entries:
        planInfo = db.query(Plans).filter(Plans.plan_id == entry.plan_id).first()
        if entry.status:
            total_active_plans+=1
            accamulated_chatbots += planInfo.total_chatbots_allowed
            accamulated_responce_tokens_credits += planInfo.Total_Responce_Tokens_allowed
            accamulated_knowledgeStore_Tokens += planInfo.total_knowldegStores_Allowed_Tokens
            accamulated_no_of_knowledge_Store += planInfo.total_knowldegeStores_allowed
            accamulated_price_paid += planInfo.price
        entry.PlanInfo = planInfo
        TrasnactionOfPlan = db.query(PaymentsTransactions).filter(PaymentsTransactions.plan_id == entry.plan_id,
                                                                  PaymentsTransactions.user_id==user_id).all()
        
        visited=set()
        uniques=[]
        for item in TrasnactionOfPlan:
         if str(item.transactionInfo) not in visited:
            visited.add(str(item.transactionInfo))
            uniques.append(item)
        #for item in TrasnactionOfPlan:
        
        TrasnactionOfPlan=uniques
        if TrasnactionOfPlan:
            entry.TransactionHistory = TrasnactionOfPlan
        else:
            entry.TransactionHistory = "Basic Plan Attached (No History)"

    aggregatedPlan={
                        "user_id":user_id,
                        "Total_Plans": len(entries),
                        "Total_active_Plans": total_active_plans,
                        "Total_inactive_Plans": len(entries) - total_active_plans,
                        "Aggregated_Chatbots":accamulated_chatbots,
                        "Aggregated_Monthly_Chatbot_Response_Tokens":accamulated_responce_tokens_credits,
                        "Aggregated_Knowledge_Store_Tokens":accamulated_knowledgeStore_Tokens ,
                        "Aggregated_Knowledge_Store":accamulated_no_of_knowledge_Store,
                        "Aggregated_Cost":accamulated_price_paid,
                    }
    userinfo = db.query(Users).filter(Users.user_id == user_id).first()
    aggreagate_consumption=db.query(Consumption).filter(Consumption.user_id == user_id).first()
    return    {
        "User_Information": userinfo,
        "Individual_Plans_Information": entries,
        "Aggregated_Plans_Information":aggregatedPlan,
        "Aggregated_Consumption_Qouta":aggreagate_consumption
    }
def updatetheConsumptionAfterPlanInACtivation(db,plan_id,user_id):
    planInfo = db.query(Plans).filter(Plans.plan_id == plan_id).first()
    consumption = db.query(Consumption).filter(Consumption.user_id == user_id).first()
    consumption.consumed_stores = max(0, consumption.consumed_stores - planInfo.total_knowldegeStores_allowed)
    consumption.consumed_chatbots = max(0, consumption.consumed_chatbots - planInfo.total_chatbots_allowed)
    consumption.consumed_store_tokens = max(0,
                                            consumption.consumed_store_tokens - planInfo.total_knowldegStores_Allowed_Tokens)
    consumption.consumed_chatbot_response_tokens = max(0,
                                                       consumption.consumed_chatbot_response_tokens - planInfo.Total_Responce_Tokens_allowed)
    consumption.last_updated = datetime.utcnow()
    db.add(consumption)
    db.add(consumption)
    db.commit()

def deactiavtePlan(subscription_id):
    db = SessionLocal()
    try:
#        canceled_subscription = stripe.Subscription.delete(subscription_id)
#        db = SessionLocal()
        entry = db.query(Selected_Plan).filter(Selected_Plan.subscription_id == subscription_id).first()
        if entry:
            if entry.status:
                entry.status=False
                entry.message="Deactivated"
                entry.last_updated=datetime.utcnow()
                db.add(entry)
                db.commit()
                updatetheConsumptionAfterPlanInACtivation(db,entry.plan_id,entry.user_id)
                db.close()
                canceled_subscription = stripe.Subscription.delete(subscription_id)
                return {"status": "ok", "message": "Given Plan set to be Inactive.", "data": None}
            else:
                raise HTTPException(status_code=404, detail="Give Plan is already Inactive.")
        else:
            raise HTTPException(status_code=404, detail="User Plan information not found under Given ID.")
    except Exception as e:
        print(e)
        db.close()
        return {"status": "error", "message": str(e), "data": None}

def deactiavtePlanAfterPeriod(subscription_id):
    db = SessionLocal()
    try:
        entry = db.query(Selected_Plan).filter(Selected_Plan.subscription_id == subscription_id).first()
        if entry:
            if entry.status:
                entry.status=False
                entry.message="Deactivated"
                entry.last_updated=datetime.utcnow()
                db.add(entry)
                db.commit()
                updatetheConsumptionAfterPlanInACtivation(db,entry.plan_id,entry.user_id)
                db.close()
                return {"status": "ok", "message": "Given Plan set to be Inactive.", "data": None}
            else:
                raise HTTPException(status_code=404, detail="Give Plan is already Inactive.")
        else:
            raise HTTPException(status_code=404, detail="User Plan information not found under Given ID.")
    except Exception as e:
        print(e)
        db.close()
        return {"status": "error", "message": str(e), "data": None}

@app.get("/MakeUserPlanInactiveByPlanPurchaseId/{subscription_ID}")
async def MakeUserPlanInactiveByPlanPurchaseId(subscription_ID: str):
    return deactiavtePlan(subscription_ID)

def deactivate_plan_at_period(subscription_id):
    try:
        db = SessionLocal()
        entry = db.query(Selected_Plan).filter(Selected_Plan.subscription_id == subscription_id).first()
        if entry:
            if entry.message=="Scheduled for Cancelation":
                raise HTTPException(status_code=404, detail="Plan is already scheduled for Cancelation.")

        updated_subscription = stripe.Subscription.modify(
            subscription_id,
            cancel_at_period_end=True
        )
#        db = SessionLocal()
#        entry = db.query(Selected_Plan).filter(Selected_Plan.subscription_id == subscription_id).first()
        if entry:
 #           if entry.message=="Scheduled for Cancelation":
  #              raise HTTPException(status_code=404, detail="Plan is already scheduled for Cancelation.")
            if entry.status:
                entry.message="Scheduled for Cancelation"
                entry.last_updated=datetime.utcnow()
                db.add(entry)
                db.commit()
                db.close()
        
        print(f"Subscription {subscription_id} is scheduled to cancel at the end of the current period.")
        return {"status": "ok", "message": f"Subscription {subscription_id} is scheduled to cancel at the end of the current period.", "data": None}
    except stripe.error.StripeError as e:
        print(f"An error occurred: {e}")
        return {"status": "error", "message": str(e), "data": None}


@app.get("/MakeUserPlanInactiveatPeriodEndByPlanPurchaseId/{subscription_ID}")
async def MakeUserPlanInactiveatPeriodEndByPlanPurchaseId(subscription_ID: str):
    return deactivate_plan_at_period(subscription_ID)

@app.get("/GetAllPlansByUserId/{user_id}")
async def GetAllPlansByUserId(user_id: str):
    try:
        db = SessionLocal()
        entries = db.query(Selected_Plan).filter(Selected_Plan.user_id==user_id).all()
        if entries:
            response=getAggreatedPlansInfoByUsersPlanList(db, user_id, entries)
            chatbots = db.query(ChatBotsConfigurations).filter(ChatBotsConfigurations.user_id==user_id).all()
            stores = db.query(knowledge_Store).filter(knowledge_Store.user_id == user_id).all()
            response['Total_Available_Chatbots']=len(chatbots)
            response['Total_Available_Stores']=len(stores)
            db.close()
            return {"status": "ok", "message": "Plans Data returned Successfully.", "data": response}
        else:
            raise HTTPException(status_code=404, detail="No Plans Found under GIven user ID.")
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}
@app.post("/AddPlanToOurPlans/")
async def AddPlanToOurPlans(request: PlansPydnaticModel):
    try:
        db = SessionLocal()
        db_entry = Plans(**request.dict())
        db.add(db_entry)
        db.commit()
        db.close()
        return {"status": "ok", "message": "Plan Added to the Our Plans successfully.", "data": None}
    except Exception as e:
            db.close()
            return {"status": "error","message": str(e), "data": None}

@app.post("/register/")
def register(request: user):
    try:
        db = SessionLocal()
        existing_user = db.query(Users).filter(Users.email == request.email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered Please use any other Email Address or Login with Existing Email Address.")
        request.password = pwd_context.hash(request.password)
        db_entry = Users(**request.dict())
        db.add(db_entry)
        db.commit()
        db.refresh(db_entry)
        db_entry.verification_token=generate_verification_token()
        db.commit()

        if request.redirected:
            db_entry.isVerified = True
            db.commit()
            db.close()
            return {"status": "ok",
                    "message": "User registered successfully.",
                    "data": db_entry}
        else:
            send_verifiaction_code_on_email(db_entry.email,db_entry.name,db_entry.verification_token)
            db.commit()
            db.close()
            return {"status": "ok",
                    "message": "User registered successfully. Check your email for verification instructions.",
                    "data": db_entry}



    except Exception as e:
            db.close()
            return {"status": "error","message": str(e), "data": None}
@app.post("/login/")
def login(email: str, password: str):
    try:
        db = SessionLocal()
        user = db.query(Users).filter(Users.email == email).first()
        if user:
            if user.isVerified:
                if not user or not pwd_context.verify(password, user.password):
                    raise HTTPException(status_code=401, detail="Invalid Credentials.")
                user_dict = {"user_id": user.user_id, "name": user.name, "email": user.email}
                db.close()
                return {"status": "ok", "message": "Account has been Authenticated.", "data": user_dict}
            else:
                send_verifiaction_code_on_email(user.email, user.name, user.verification_token)
                return {"status": "ok", "message": "Your account is not Verified Please Check Your Email Address for Verification Link.", "data": None}
        else:
            return {"status": "error",
                    "message": "User not Found.",
                    "data": None}
    except Exception as e:
            db.close()
            return {"status": "error","message": str(e), "data": None}
@app.post("/forgot_password/")
async def forgot_password(email: str):
    try:
        db = SessionLocal()
        user = db.query(Users).filter(Users.email == email).first()
        if user:
            send_password_reset(user.email, user.name, user.verification_token)
            db.commit()
            db.close()
            return {"status": "ok", "message": "If the email you provided is associated with an account you will receive a new confirmation to activate your account.", "data": None}
        else:
            return {"status": "error",
                    "message": "User not Found.",
                    "data": None}
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}
@app.post("/reset_password/")
async def reset_password(request: passwordreset):
    try:
        db = SessionLocal()
        user = db.query(Users).filter(Users.verification_token == request.token).first()
        if user:
            user.password=pwd_context.hash(request.new_password)
            db.commit()
            db.close()
            return {"status": "ok",
                    "message": "password has been Reset Successfully..",
                    "data": None}
        else:
            return {"status": "error",
                    "message": "User not Found.",
                    "data": None}
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}
def generate_verification_token():
    return secrets.token_urlsafe(16)
def send_verifiaction_code_on_email(receiver,good_name,verification_token):
    port = 587  # For starttls
    smtp_server = "cloudhost-4480981.us-midwest-1.nxcli.net"
    sender_email = "support@optimalaccess.com"
    receiver_email = receiver
    subject = "Account Verification Link (KChat powered by Optimal Access)"
    password = "StakedMethodRoodTannin"

    body="""<!DOCTYPE html>
    <html>
      <head>
        <style>
          * {
            font-family: "Montserrat", sans-serif;
            color:white;
          }
          ul li
            {
                margin-bottom:5px;
            }
          body {
            font-family: Arial, sans-serif;
            background-color: #2B3AA4;
            color: white;
            font-family: "Montserrat", sans-serif;
            padding:20px;
          }
          .container {
            max-width: 100%;
            color: white;
            margin: 0 auto;
            padding: 20px;
            border: 1px solid white;
            background-color: #2B3AA4;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
          }

          .footer-like {
            margin-top: auto;
            padding: 6px;
            text-align: center;
          }
          .footer-like p {
            margin: 0;
            padding: 4px;
            color: #fafafa;
            font-family: "Raleway", sans-serif;
            letter-spacing: 1px;
          }
          .footer-like p a {
            text-decoration: none;
            font-weight: 600;
          }

          .logo {
            width: 100px;
            border:1px solid white;
          }
          .verify-button
          {
          background-color:white;
          border-radius:5px;
          padding:10px;
          border: none;
          text-decoration:none;
          }
        </style>
      </head>
      <body>
        <div class="container">
    <img src="https://i.ibb.co/qxxjqcZ/Untitled-design-removebg-preview-1.png" alt="Optimal-Access-Logo" border="0" class="logo" />
    """
    body += f'<p>Dear {good_name},</p>' \
            f'<h1><strong>Welcome to kChat!</strong></h1>' \
            f'<p>Your Account Verification Link is placed below. Click on the link to get verified:</p>' \
            f'<h4><b><a href="{BASE_URL}verify?token={verification_token}" class="verify-button">Click here to Verify Your Account</a></b></h4>'
    body+="""
          <p><b>Sincerely,</b><br />The KChat Team</p>
          <div class="footer-like">
            <p>
              Powered by Optimal Access
            </p>
          </div>
        </div>
      </body>
    </html>"""
    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    # Add body to email
    message.attach(MIMEText(body, "html"))

    # Create a secure SSL context
    context = ssl.create_default_context()

    # Try to log in to server and send email
    try:
        server = smtplib.SMTP(smtp_server, port)
        server.starttls(context=context)  # Secure the connection
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())
    except Exception as e:
        print(f"Failed to send email: {e}")
    finally:
        server.quit()
def send_password_reset(receiver,good_name,verification_token):
    port = 587  # For starttls
    smtp_server = "cloudhost-4480981.us-midwest-1.nxcli.net"
    sender_email = "support@optimalaccess.com"
    receiver_email = receiver
    subject = "Password Reset Link (KChat powered by Optimal Access)"
    password = "StakedMethodRoodTannin"

    body="""<!DOCTYPE html>
    <html>
      <head>
        <style>
          * {
            font-family: "Montserrat", sans-serif;
            color:white;
          }
          ul li
            {
                margin-bottom:5px;
            }
          body {
            font-family: Arial, sans-serif;
            background-color: #2B3AA4;
            color: white;
            font-family: "Montserrat", sans-serif;
            padding:20px;
          }
          .container {
            max-width: 100%;
            color: white;
            margin: 0 auto;
            padding: 20px;
            border: 1px solid white;
            background-color: #2B3AA4;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
          }

          .footer-like {
            margin-top: auto;
            padding: 6px;
            text-align: center;
          }
          .footer-like p {
            margin: 0;
            padding: 4px;
            color: #fafafa;
            font-family: "Raleway", sans-serif;
            letter-spacing: 1px;
          }
          .footer-like p a {
            text-decoration: none;
            font-weight: 600;
          }

          .logo {
            width: 100px;
            border:1px solid white;
          }
          .verify-button
          {
          background-color:white;
          border-radius:5px;
          padding:10px;
          border: none;
          text-decoration:none;
          }
        </style>
      </head>
      <body>
        <div class="container">
    <img src="https://i.ibb.co/qxxjqcZ/Untitled-design-removebg-preview-1.png" alt="Optimal-Access-Logo" border="0" class="logo" />
    """
    body += f'<h1>Hello, {good_name},</h1>' \
            f'<p>A request has been received to change the password for your Kchat account.</p>' \
            f'<h4><b><a href="https://kchat.website/auth/password_reset?token={verification_token}" class="verify-button">Click here to Reset Password</a></b></h4>'
    body+="""
          <p><b>Sincerely,</b><br />The KChat Team</p>
          <div class="footer-like">
            <p>
              Powered by Optimal Access
            </p>
          </div>
        </div>
      </body>
    </html>"""
    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    # Add body to email
    message.attach(MIMEText(body, "html"))

    # Create a secure SSL context
    context = ssl.create_default_context()

    # Try to log in to server and send email
    try:
        server = smtplib.SMTP(smtp_server, port)
        server.starttls(context=context)  # Secure the connection
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())
    except Exception as e:
        print(f"Failed to send email: {e}")
    finally:
        server.quit()
@app.get("/resendVerificationToken/")
def resendVerificationToken(email: str):
    try:
        db = SessionLocal()
        user = db.query(Users).filter(Users.email == email).first()
        if user:
            if user.isVerified:
                db.close()
                return {"status": "ok", "message": "Account is already Verified.", "data": None}
            else:
                send_verifiaction_code_on_email(user.email, user.name, user.verification_token)
                return {"status": "ok", "message": "Verification Link has been Resent to your Email Address.", "data": None}
        else:
            return {"status": "error",
                    "message": "Provided Email Address does not points to any Registered account.",
                    "data": None}
    except Exception as e:
            db.close()
            return {"status": "error","message": str(e), "data": None}
def generate_verification_code():
    # Generate a random 6-digit hexadecimal code
    verification_code = secrets.token_hex(3).upper()
    return verification_code
@app.get("/verify/")
def verify_account(token: str):
    try:
        db = SessionLocal()
        user = db.query(Users).filter(Users.verification_token == token).first()
        if user:
            if user.isVerified:
                db.close()
                return RedirectResponse(url="https://kchat.website/auth/signin/?message=Account+is+already+Verified.")  # Redirect to the given page with a message
                #return {"status": "ok", "message": "Account is already Verified.", "data": None}
            else:
                user.isVerified=True
                db.commit()
                db.close()
                return RedirectResponse(url="https://kchat.website/auth/signin/?message=Your+account+has+been+verified+login+to+access+Dashboard.")  # Redirect to the given page with a message
        else:
            return {"status": "error",
                    "message": "User not Found.",
                    "data": None}
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}
@app.post("/SendVerificationEmail/")
async def SendVerificationEmail (receiver:str):
    send_verifiaction_code_on_email(receiver,"Ali Haider","TestToken")
def upadteChatlogs(db,chatbot_id,visitor_ID,Human_Message,AI_Responce,context):
    ChatLogs_Object = ChatLogs(chatbot_id=chatbot_id, visitor_ID=visitor_ID,
                               Human_Message=Human_Message, AI_Responce=AI_Responce,context=context)
    db.add(ChatLogs_Object)
    db.commit()
    return
@app.post("/Chat/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        db = SessionLocal()
        db_entry = db.query(ChatBotsConfigurations).filter(ChatBotsConfigurations.chatbot_id == request.chatbotId).first()

        if db_entry:
        #    print(db_entry.knowledgeBases,db_entry.temperature,db_entry.llm)
            if db_entry.knowledgeBases == "[]":
                raise HTTPException(status_code=404,
                                    detail=f'Chatbot Do not have any Knowledge Store attached to it Please Contact Administrator to Attach Context.')
            try:
                answer,sources=Get_Conversation_chain(db_entry.knowledgeBases,db_entry.temperature,db_entry.llm,request.question,request.chat_history)
#                print(answer,sources)
                currentTokens=num_tokens_from_string(str(request.question) + str(answer) + str(sources))
                verification = VerifyChatbotResponceCreditQouta(db, db_entry.user_id, 1)
                if not verification[1]:
                    raise HTTPException(status_code=404, detail=f'{verification[0]}')
                accepted_keys=ast.literal_eval(db_entry.OutBoundLinks)
                accepted_keys.append("HeadLine")
                if db_entry.AddContext==1:
                    accepted_keys.append("Context-Information")
                # Filter out dictionaries based on accepted keys

                filtered_sources_list = [dict(filter(lambda item: item[0] in accepted_keys, d.items())) for d in sources]
                upadteChatlogs(db,db_entry.chatbot_id,request.visitorID,request.question,answer,str(filtered_sources_list))
                updateChatbotsResponseCreditCount(db, db_entry.user_id,1)
                db.close()
                return ChatResponse(answer=answer, reference_context=filtered_sources_list)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        else:
            db.close()
            return ChatResponse(answer="Chatbot Configuration not Found under ID: " + str(request.chatbotId) , reference_context=[])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
def num_tokens_from_string(string: str, encoding_name="text-embedding-ada-002") -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
def send_Repsonse_credicts_exhaust_Mail(receiver,good_name,consumed_percentage):
    port = 587  # For starttls
    smtp_server = "cloudhost-4480981.us-midwest-1.nxcli.net"
    sender_email = "support@optimalaccess.com"
    receiver_email = receiver
    subject = "Response Credits Exhausted (KChat powered by Optimal Access)"
    password = "StakedMethodRoodTannin"

    body="""<!DOCTYPE html>
    <html>
      <head>
        <style>
          * {
            font-family: "Montserrat", sans-serif;
            color:white;
          }
          ul li
            {
                margin-bottom:5px;
            }
          body {
            font-family: Arial, sans-serif;
            background-color: #2B3AA4;
            color: white;
            font-family: "Montserrat", sans-serif;
            padding:20px;
          }
          .container {
            max-width: 100%;
            color: white;
            margin: 0 auto;
            padding: 20px;
            border: 1px solid white;
            background-color: #2B3AA4;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
          }

          .footer-like {
            margin-top: auto;
            padding: 6px;
            text-align: center;
          }
          .footer-like p {
            margin: 0;
            padding: 4px;
            color: #fafafa;
            font-family: "Raleway", sans-serif;
            letter-spacing: 1px;
          }
          .footer-like p a {
            text-decoration: none;
            font-weight: 600;
          }

          .logo {
            width: 100px;
            border:1px solid white;
          }
          .verify-button
          {
          background-color:white;
          border-radius:5px;
          padding:10px;
          border: none;
          text-decoration:none;
          }
        </style>
      </head>
      <body>
        <div class="container">
    <img src="https://i.ibb.co/qxxjqcZ/Untitled-design-removebg-preview-1.png" alt="Optimal-Access-Logo" border="0" class="logo" />
    """
    body += f'<h1>Hello, {good_name},</h1>' \
            f'<p>You chatbots had Consumed {consumed_percentage}% of your Response Credits.</p>' \
            f'<p>We are writing to inform you that you have reached your chatbot response credits limit. To ensure the continued operation of your Chatbot Service without any interruptions, it is imperative to renew your plan or purchase additional credits immediately.\n\nThank you for your prompt attention to this matter.</p>'
    body+="""
          <p><b>Sincerely,</b><br />The KChat Team</p>
          <div class="footer-like">
            <p>
              Powered by Optimal Access
            </p>
          </div>
        </div>
      </body>
    </html>"""
    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    # Add body to email
    message.attach(MIMEText(body, "html"))

    # Create a secure SSL context
    context = ssl.create_default_context()

    # Try to log in to server and send email
    try:
        server = smtplib.SMTP(smtp_server, port)
        server.starttls(context=context)  # Secure the connection
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())
    except Exception as e:
        print(f"Failed to send email: {e}")
    finally:
        server.quit()
def VerifyChatbotResponceCreditQouta(db,user_id,currentTokens):
    Aggregated_Plans_Information,Aggregated_Consumption=getAggreatedPlansConsumptionByUsersPlanList(db,user_id)
    reamining_chatbots_responces=Aggregated_Plans_Information['Aggregated_Monthly_Chatbot_Response_Tokens']-Aggregated_Consumption.consumed_chatbot_response_tokens
    if Aggregated_Plans_Information['Aggregated_Monthly_Chatbot_Response_Tokens']>0:
        percentage_consumed=round((Aggregated_Consumption.consumed_chatbot_response_tokens/Aggregated_Plans_Information['Aggregated_Monthly_Chatbot_Response_Tokens'])*100,2)
        user = db.query(Users).filter(Users.user_id == user_id).first()
        if percentage_consumed>80:
            send_Repsonse_credicts_exhaust_Mail(user.email,user.name,percentage_consumed)
        if reamining_chatbots_responces<=0:
            return f"Chatbots Response Credits Quota Exceeds (Response Quota Left {max(reamining_chatbots_responces,0)} ) Upgrade Plan for More Credits",False
        return "", True
    else:
        return "Please Add any plan you donot have any active plan.",False
def updateChatbotsResponseCreditCount(db,user_id,curremt_demanded_token):
    comsuptionObj = db.query(Consumption).filter(Consumption.user_id == user_id).first()
    comsuptionObj.consumed_chatbot_response_tokens=comsuptionObj.consumed_chatbot_response_tokens+curremt_demanded_token
    comsuptionObj.last_updated=datetime.utcnow()
    db.add(comsuptionObj)
    db.commit()
def VerifyChatbotCreationQouta(db,user_id):
    Aggregated_Plans_Information,Aggregated_Consumption=getAggreatedPlansConsumptionByUsersPlanList(db,user_id)
    reamining_chatbots=Aggregated_Plans_Information['Aggregated_Chatbots']-Aggregated_Consumption.consumed_chatbots
    if reamining_chatbots<=0:
        return f"Chatbots Creation Quota Exceeds (Quota Left for Creation {max(reamining_chatbots,0)} )",False
    return "", True
def updateChatbotsCreationCount(db,user_id):
    comsuptionObj = db.query(Consumption).filter(Consumption.user_id == user_id).first()
    comsuptionObj.consumed_chatbots=comsuptionObj.consumed_chatbots+1
    db.add(comsuptionObj)
    db.commit()
@app.delete("/DeleteChatLogByChatID/{chatbot_id}")
async def DeleteChatLogByChatID(chatbot_id: str):
    db = SessionLocal()
    try:
        ChatLogs_info = db.query(ChatLogs).filter(ChatLogs.chatbot_id == chatbot_id)
        if ChatLogs_info.count() > 0:
            ChatLogs_info.delete()
            db.commit()
            db.close()
            return {"status": "ok", "message": "Chatbot's Chat Logs deleted successfully.", "data": None}
        else:
            raise HTTPException(status_code=404, detail="Chatbot ChatLogs not Found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
@app.get("/GetChatLogsByChatBotID/{chatbot_id}")
async def GetChatLogsByChatBotID(chatbot_id: str):
    try:
        db = SessionLocal()
        chatlogs = db.query(ChatLogs).filter(ChatLogs.chatbot_id == chatbot_id).order_by(ChatLogs.responded_at.desc()).all()
        if not chatlogs:
            raise HTTPException(status_code=404, detail="Chatbot do not Contains any Chat Logs.")
        db.close()
        return {"status": "ok", "message": "Chatbots Chat Logs returned Successfully.", "data": chatlogs}
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}
@app.get("/GetChatLogsByUserID/{user_id}")
async def GetChatLogsByUserID(user_id: str):
    try:
        db = SessionLocal()
        chatbots = db.query(ChatBotsConfigurations).filter(ChatBotsConfigurations.user_id == user_id).all()
        ids=[]
        for bot in chatbots:
            ids.append(bot.chatbot_id)

        chatlogs = db.query(ChatLogs).filter(ChatLogs.chatbot_id.in_(ids)).order_by(ChatLogs.responded_at.desc()).all()
        if not chatlogs:
            raise HTTPException(status_code=404, detail="User Chatbot do not Contains any Chat Logs.")
        db.close()
        return {"status": "ok", "message": "All Chatbots Chat Logs under given USer ID returned Successfully.", "data": chatlogs}
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}
def updateKnowledgeBaseEditandTokensCount(db,user_id,tokens_count):
    comsuptionObj = db.query(Consumption).filter(Consumption.user_id == user_id).first()
    comsuptionObj.consumed_store_tokens=comsuptionObj.consumed_store_tokens+tokens_count
    comsuptionObj.last_updated=datetime.utcnow()
    db.add(comsuptionObj)
    db.commit()
def VerifyKnowldegeBaseEditQouta(db,user_id,current_tokens_demanded):
    Aggregated_Plans_Information,Aggregated_Consumption=getAggreatedPlansConsumptionByUsersPlanList(db,user_id)
    reamining_knowledge_base = Aggregated_Plans_Information['Aggregated_Knowledge_Store'] - Aggregated_Consumption.consumed_stores
    reamining_knowledge_base_tokens = Aggregated_Plans_Information['Aggregated_Knowledge_Store_Tokens'] - Aggregated_Consumption.consumed_store_tokens-current_tokens_demanded
    responceText=""
    verified=True
    if current_tokens_demanded>reamining_knowledge_base_tokens:
        responceText+=f"Requested Knowledge Base Tokens Quota Exceeds Try Source with less Number of Tokens.\n"
        verified=False

    if reamining_knowledge_base_tokens<=0:
        responceText +=f"Knowledge Base Tokens Quota Exceeds (Remaining Quota {max(reamining_knowledge_base_tokens,0)} Tokens) update the Plan and Try Again.\n"
        verified=False
    return responceText, verified
    
# Endpoint to create a new entry
@app.post("/EditKnowledgeStore/")
def EditKnowledgeStore(entry: knowledgeStoreEdit):
    db = SessionLocal()
    db_entry=None
    latest_knowlegde_id=None
    try:
        if not TestAPISQouta():
            raise HTTPException(status_code=404, detail=f'Knowledge Store Embeddings API Qouta Exceeds Please Contact Administrator.')

        knowledge_store = db.query(knowledge_Store).filter(knowledge_Store.user_id == entry.user_id,knowledge_Store.knowledge_base_id==entry.knowledge_base_id).first()
        if not knowledge_store:
            raise HTTPException(status_code=404, detail="Knowledge store not found under Given ID")
        docs, tokenscount = GetDocumentsFromURL(entry.user_id, str(knowledge_store.knowledge_base_id), entry.xml_url,entry.wordpress_base_url)
        if docs != False and tokenscount!=-1:
            verification = VerifyKnowldegeBaseEditQouta(db, entry.user_id, tokenscount)
            if not verification[1]:
                raise HTTPException(status_code=404, detail=f'{verification[0]}')
            if deleteVectorsusingKnowledgeBaseID(entry.knowledge_base_id):
                knowledge_store.descriptive_name=entry.descriptive_name
                knowledge_store.xml_url=entry.xml_url
                knowledge_store.wordpress_base_url=entry.wordpress_base_url
                knowledge_store.syncing_feature=entry.syncing_feature
                knowledge_store.syncing_period=entry.syncing_period
                knowledge_store.syncing_state=entry.syncing_state
                db.commit()
                db.refresh(knowledge_store)
                
                if create_chatbot(docs):
                    db.close()
                    updateKnowledgeBaseEditandTokensCount(db, entry.user_id, tokenscount)
                    return {"status": "ok", "message": "Knowledge Store Successfully Updated.",
                            "data": {"Token_Tokens_consumed",tokenscount}}
            else:
                raise HTTPException(status_code=404, detail=f'Some error Accured while Updating the Knowledge Store.')
                
    except Exception as e:
        db.close()
        return {"status": "error","message": str(e), "data": None}
        
@app.post("/KnowledgeStore/")
def create_knowledge_Store(entry: knowledgeStoreCreate):
    db = SessionLocal()
    db_entry=None
    latest_knowlegde_id=None
    try:
        if not TestAPISQouta():
            raise HTTPException(status_code=404, detail=f'Knowledge Store Embeddings API Qouta Exceeds Please Contact Administrator.')

        db_entry = knowledge_Store(**entry.dict())
        db.add(db_entry)
        db.commit()
        db.refresh(db_entry)
        latest_knowlegde_id = db_entry.knowledge_base_id  # Get the knowledge base ID of the latest entry
        docs, tokenscount = GetDocumentsFromURL(entry.user_id, str(latest_knowlegde_id), entry.xml_url,entry.wordpress_base_url)
       # print(tokenscount,len(docs))
        if docs != False and tokenscount!=-1:
            verification = VerifyKnowldegeBaseCreationQouta(db, entry.user_id, tokenscount)
            if not verification[1]:
                raise HTTPException(status_code=404, detail=f'{verification[0]}')
            if create_chatbot(docs):
                db.close()
                updateKnowledgeBaseCreationandTokensCount(db, entry.user_id, tokenscount)
                return {"status": "ok", "message": "Knowledge Store Successfully Created.",
                        "data": {"Token_Tokens_consumed",tokenscount}}
    except Exception as e:
        last_entry = db.query(knowledge_Store).filter(knowledge_Store.knowledge_base_id == db_entry.knowledge_base_id).first()
        if last_entry:
            db.delete(last_entry)
            db.commit()
        db.close()
        return {"status": "error","message": str(e), "data": None}
def VerifyKnowldegeBaseCreationQouta(db,user_id,current_tokens_demanded):
    Aggregated_Plans_Information,Aggregated_Consumption=getAggreatedPlansConsumptionByUsersPlanList(db,user_id)
    reamining_knowledge_base = Aggregated_Plans_Information['Aggregated_Knowledge_Store'] - Aggregated_Consumption.consumed_stores
    reamining_knowledge_base_tokens = Aggregated_Plans_Information['Aggregated_Knowledge_Store_Tokens'] - Aggregated_Consumption.consumed_store_tokens-current_tokens_demanded
    responceText=""
    verified=True
    if current_tokens_demanded>reamining_knowledge_base_tokens:
        responceText+=f"Requested Knowledge Base Tokens Quota Exceeds Try Source with less Number of Tokens.\n"
        verified=False
    if reamining_knowledge_base<=0:
        responceText +=f"Knowledge Base Creation Quota Exceeds (Remaining Quota {max(reamining_knowledge_base,0)} ) update the Plan and Try Again.\n"
        verified=False

    if reamining_knowledge_base_tokens<=0:
        responceText +=f"Knowledge Base Tokens Quota Exceeds (Remaining Quota {max(reamining_knowledge_base_tokens,0)} Tokens) update the Plan and Try Again.\n"
        verified=False
    return responceText, verified
def updateKnowledgeBaseCreationandTokensCount(db,user_id,tokens_count):
    comsuptionObj = db.query(Consumption).filter(Consumption.user_id == user_id).first()
    comsuptionObj.consumed_stores=comsuptionObj.consumed_stores+1
    comsuptionObj.consumed_store_tokens=comsuptionObj.consumed_store_tokens+tokens_count
    comsuptionObj.last_updated=datetime.utcnow()
    db.add(comsuptionObj)
    db.commit()

@app.post("/CreateChatbots/")
def createChatbot(entry: ChatBots):
    db_entry=None
    appaeranceEntry=None
    db = SessionLocal()
    try:
        verification = VerifyChatbotCreationQouta(db, entry.user_id)
        if not verification[1]:
            raise HTTPException(status_code=404, detail=f'{verification[0]}')

        db_entry = ChatBotsConfigurations(**entry.dict())
        db.add(db_entry)
        db.commit()
        db.refresh(db_entry)
        appaeranceEntry=ChatbotAppearnace(chatbot_id=db_entry.chatbot_id,ThemeColor="#4a57b4",InitialMessage=f"I am {db_entry.descriptive_name}, how can I help you?",DisplayName=db_entry.descriptive_name)
        db.add(appaeranceEntry)
        db.commit()
        updateChatbotsCreationCount(db,entry.user_id)
        db.close()
        return {"status": "ok", "message": "Chatbot Configuration stored successfully.", "data": None}
    except Exception as e:
        if  db_entry and appaeranceEntry:
            last_entry = db.query(ChatBotsConfigurations).filter(ChatBotsConfigurations.chatbot_id == db_entry.chatbot_id).first()
            if last_entry:
                db.delete(last_entry)
                db.commit()
            last_entry = db.query(ChatbotAppearnace).filter(ChatbotAppearnace.chatbot_id == appaeranceEntry.chatbot_id).first()
            if last_entry:
                db.delete(last_entry)
                db.commit()
            db.close()

        return {"status": "error", "message": str(e), "data": None}
@app.get("/GetChatBotApprancebyChatbotID/{chatbot_id}")
async def GetChatBotApprancebyChatbotID(chatbot_id: str):
    try:
        db = SessionLocal()
        chatbot = db.query(ChatbotAppearnace).filter(ChatbotAppearnace.chatbot_id == chatbot_id).first()
        if not chatbot:
            raise HTTPException(status_code=404, detail="Bot Appearance not found")
        db.close()
        return {"status": "ok", "message": "Appearance Returned Successfully.", "data": chatbot}
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}
@app.put("/EditChatbotAppearance/{chatbot_id}")
def EditChatbotAppearance(chatbot_id: str, edited_Appearance_info: EditAppeanceChatBots):
    try:
        db = SessionLocal()
        db_entry = db.query(ChatbotAppearnace).filter(ChatbotAppearnace.chatbot_id == chatbot_id).first()
        if db_entry is None:
            raise HTTPException(status_code=404, detail="Chatbot Apperance Configuration not Found")

        for key, value in edited_Appearance_info.dict().items():
            setattr(db_entry, key, value)
        db.commit()
        db.refresh(db_entry)
        db.close()
        return {"status": "ok", "message": "Chatbot information updated successfully.", "data": None}
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}
@app.put("/EditChatbot/{chatbot_id}")
def edit_chatbot(chatbot_id: str, edited_info: EditChatBots):
    try:
        db = SessionLocal()
        db_entry = db.query(ChatBotsConfigurations).filter(ChatBotsConfigurations.chatbot_id == chatbot_id).first()
        if db_entry is None:
            raise HTTPException(status_code=404, detail="Chatbot Configuration not Found")

        for key, value in edited_info.dict().items():
            setattr(db_entry, key, value)
        db.commit()
        db.refresh(db_entry)
        db.close()
        return {"status": "ok", "message": "Chatbot information updated successfully.", "data": None}
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}
@app.post("/TurnSycningFeatureOn/")
def TurnSyncingFeatureOn(entry: syncingFeatureStatus):
    try:
        db = SessionLocal()
        knowldegeStore = db.query(knowledge_Store).filter(knowledge_Store.knowledge_base_id == entry.knowledgeStoreId).first()
        if knowldegeStore:
            if knowldegeStore.syncing_feature==1:
                db.close()
                return {"status": "ok", "message": "Syncing Feature already Turned on for Vector Storage with ID: " + str(
                    entry.knowledgeStoreId), "data": None}
            knowldegeStore.syncing_feature = 1
            knowldegeStore.syncing_period=entry.syncPeriod
            db.commit()
            db.close()
            return {"status": "ok", "message": "Syncing Feature Successfully Turned on for Vector Storage with ID: " + str(
                entry.knowledgeStoreId), "data": None}
        else:
            db.close()
            raise HTTPException(status_code=404, detail="Knowledge base with ID: " + str(entry.knowledgeStoreId) + " not found.")
    except Exception as e:
        return {"status": "error", "message": str(e), "data": None}
@app.get("/TurnSycningFeatureOff/{knowledge_base_id}")
def TrunSycingFeatureOff(knowledge_base_id: str):
    try:
        db = SessionLocal()
        knowldegeStore = db.query(knowledge_Store).filter(knowledge_Store.knowledge_base_id == knowledge_base_id).first()
        if knowldegeStore:
            if knowldegeStore.syncing_feature==0:
                db.close()
                return {"status": "ok",
                        "message": "Syncing Feature already Turned Off for Vector Storage with ID: " + str(
                            knowledge_base_id), "data": None}

            knowldegeStore.syncing_feature = 0
            knowldegeStore.syncing_state=0
            knowldegeStore.syncing_period=0
            job = db.query(syncingJob).filter(syncingJob.knowledge_base_id == knowledge_base_id).first()
            if job:
                db.delete(job)
            db.commit()
            db.close()
            return {"status": "ok",
                    "message": "Syncing Feature Successfully Turned Off for Vector Storage with ID: " + str(
                        knowledge_base_id), "data": None}
        else:
            db.close()
            raise HTTPException(status_code=404, detail="Knowledge base with ID: " + str(knowledge_base_id) + " not found.")
    except Exception as e:
        return {"status": "error", "message": str(e), "data": None}
@app.get("/GetChatbotsbyUserID/{user_id}")
def get_chatbots_by_user_ID(user_id: str):
    try:
        db = SessionLocal()
        entries = db.query(ChatBotsConfigurations).filter(ChatBotsConfigurations.user_id == user_id).all()
        if entries:
            for entry in entries:
                List_of_stores=[]
        #        print(entry.knowledgeBases)
                for Store in ast.literal_eval(entry.knowledgeBases):
                    storedata = db.query(knowledge_Store).filter(knowledge_Store.knowledge_base_id ==Store).first()
                    List_of_stores.append(storedata)
                entry.knowledgeBases=List_of_stores
            db.close()
            return {"status": "ok", "message": "ChatBots Configurations returned Successfully.", "data": entries}
        else:
            raise HTTPException(status_code=404, detail="Chatbots Configuration under User ID ("+str(user_id)+") not found.")
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}
@app.get("/GetChatbotEmbedableScript/{chatbot_id}")
def get_chatbots_Embdeding_Script(chatbot_id: str):
    try:
        db = SessionLocal()
        entry = db.query(ChatBotsConfigurations).filter(ChatBotsConfigurations.chatbot_id == chatbot_id).first()
        if entry:
            script= f"""<script src='https://bot.kchat.website/Chatbot.js'></script><script>setupChatbot("{entry.chatbot_id}");</script>"""
            return {"status": "ok", "message": "Chatbot Embed able SCript returned Successfully.", "data": {"script":script}}
        db.close()
        raise HTTPException(status_code=404, detail="Chatbot with ID: " + str(chatbot_id) + " not found")
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}
@app.get("/GetknowledgeStoresbyUserID/{user_id}")
def get_knowledge_store_by_user_ID(user_id: str):
    try:
        db = SessionLocal()
        entries = db.query(knowledge_Store).filter(knowledge_Store.user_id == user_id).all()
        if entries:
            db.close()
            return {"status": "ok", "message": "Knowledge Store Data returned Successfully.", "data": entries}
        else:
            raise HTTPException(status_code=404, detail="Knowledge Stores under User ID ("+str(user_id)+") not found.")
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}
@app.get("/StartSyncing/{knowledge_base_id}")
def StartSyncing(knowledge_base_id: str):
    try:
        db = SessionLocal()
        entry = db.query(knowledge_Store).filter(knowledge_Store.knowledge_base_id == knowledge_base_id).first()
        if entry:
            if entry.syncing_state==1:
                db.close()
                return {"status": "ok", "message": "Syncing already started for knowledge base with ID: " + str(
                    knowledge_base_id), "data": None}
            entry.syncing_state = 1
            job=syncingJob(
                    user_id=entry.user_id,
                    knowledge_base_id= knowledge_base_id,
                    syncing_period= entry.syncing_period,
                    xml_url= entry.xml_url,
                    wordpress_base_url=entry.wordpress_base_url,
                    last_performed=datetime.now()
            )
            db.add(job)
            db.commit()
            db.close()
            return {"status": "ok", "message": "Syncing started for knowledge base with ID: " + str(
                knowledge_base_id), "data": None}
        else:
            raise HTTPException(status_code=404, detail="Knowledge base with ID: " + str(knowledge_base_id) + " not found")
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}
@app.get("/StopSyncing/{knowledge_base_id}")
def StopSyncing(knowledge_base_id: str):
    try:
        db = SessionLocal()
        entry = db.query(knowledge_Store).filter(knowledge_Store.knowledge_base_id == knowledge_base_id).first()
        if entry:
            if entry.syncing_state==0:
                db.close()
                return {"status": "ok", "message": "Syncing already Stopped for knowledge base with ID: " + str(
                    knowledge_base_id), "data": None}
            entry.syncing_state = 0
            job = db.query(syncingJob).filter(syncingJob.knowledge_base_id == knowledge_base_id).first()
            if job:
                db.delete(job)
            db.commit()
            db.close()
            return {"status": "ok", "message": "Syncing stopped for knowledge base with ID: " + str(
                knowledge_base_id), "data": None}
        else:
            raise HTTPException(status_code=404, detail="Knowledge base with ID: " + str(knowledge_base_id) + " not found")
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}
@app.delete("/deleteChatbot/{chatbot_id}")
def delete_Chatbot(chatbot_id: str):
    try:
        db = SessionLocal()
        entry = db.query(ChatBotsConfigurations).filter(ChatBotsConfigurations.chatbot_id == chatbot_id).first()
        chatbot_appeanrce = db.query(ChatbotAppearnace).filter(ChatbotAppearnace.chatbot_id == chatbot_id).first()
        if entry:
            db.delete(entry)
            db.commit()
            db.close()
            db.delete(chatbot_appeanrce)
            db.commit()
            db.close()
            return {"status": "ok", "message": "Chatbot Deleted Successfully with ID:: " + str(
                chatbot_id), "data": None}
        else:
            raise HTTPException(status_code=404,
                                    detail="Chatbot with ID: " + str(chatbot_id) + " not found in Databse.")
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}
@app.delete("/deleteKnowledgeBase/{knowledge_base_id}")
def delete_knowledge_base(knowledge_base_id: str):
    try:
        db = SessionLocal()
        entry = db.query(knowledge_Store).filter(knowledge_Store.knowledge_base_id == knowledge_base_id).first()
        if entry:
            if deleteVectorsusingKnowledgeBaseID(knowledge_base_id):
                job = db.query(syncingJob).filter(syncingJob.knowledge_base_id == knowledge_base_id).first()
                if job:
                    db.delete(job)
                    db.commit()
                chatbots = db.query(ChatBotsConfigurations).all()
                for chatbot in chatbots:
                    knowledge_bases = json.loads(chatbot.knowledgeBases)
                    if knowledge_base_id in knowledge_bases:
                        if len(knowledge_bases) == 1:
                            db.delete(chatbot)
                            db.commit()
                        else:
                            knowledge_bases.remove(knowledge_base_id)
                            chatbot.knowledgeBases = json.dumps(knowledge_bases)
                            db.commit()
                db.delete(entry)
                db.commit()
                db.close()
                return {"status": "ok", "message": "Knowledge Base Deleted Successfully with ID:: " + str(
                    knowledge_base_id), "data": None}
            else:
                db.delete(entry)
                db.commit()
                db.close()
                return {"status": "ok", "message": "Knowledge Base Deleted Successfully with ID:: " + str(
                    knowledge_base_id), "data": None}
#raise HTTPException(status_code=404,
#                                    detail="Knowledge base with ID: " + str(knowledge_base_id) + " not found in Vector Documents.")
        else:
            raise HTTPException(status_code=404, detail="Knowledge base with ID: " + str(knowledge_base_id) + " not found")

    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}
@app.post("/AddLeadsDataToChatBot/")
async def AddLeadsDataToChatBot(entry: AddLeadsPydanticModel):
    db=None
    try:
        db = SessionLocal()
        db_lead = LeadsGenerated(**entry.dict())
        db.add(db_lead)
        db.commit()
        db.refresh(db_lead)
        db.close()
        return {"status": "ok", "message": "Lead Successfully Added to the Chatbot.", "data": None}
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}
@app.get("/GetLeadsGeneratedByChatBot/{chatbot_id}")
async def GetLeadsGeneratedByChatBot(chatbot_id: str):
    try:
        db = SessionLocal()
        leads_info = db.query(LeadsGenerated).filter(LeadsGenerated.chatbot_id == chatbot_id).all()
        if not leads_info:
            raise HTTPException(status_code=404, detail="Chatbot do not Contains any Leads.")
        db.close()
        return {"status": "ok", "message": "Chatbots Leads returned Successfully.", "data": leads_info}
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}
@app.get("/GetLeadsGeneratedByUserID/{user_id}")
async def GetLeadsGeneratedByUserID(user_id: str):
    try:
        db = SessionLocal()
        chatbots = db.query(ChatBotsConfigurations).filter(ChatBotsConfigurations.user_id == user_id).all()
        ids = []
        for bot in chatbots:
            ids.append(bot.chatbot_id)
        leads_info = db.query(LeadsGenerated).filter(LeadsGenerated.chatbot_id.in_(ids)).all()
        if not leads_info:
            raise HTTPException(status_code=404, detail="User Chatbots do not Contains any Leads.")
        db.close()
        return {"status": "ok", "message": "Chatbots Leads under User ID returned Successfully.", "data": leads_info}
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}
@app.get("/GetKnowldegebaseDashboard/{user_id}")
async def GetKnowldegebaseDashboard(user_id: str):
    try:
        db = SessionLocal()
        knowledge_stores = db.query(knowledge_Store).filter(knowledge_Store.user_id == user_id).all()
        if not knowledge_stores:
            raise HTTPException(status_code=404, detail="Knowledge store not found under user ID")
        total_knowledge_bases = db.query(func.count(knowledge_Store.knowledge_base_id).filter(knowledge_Store.user_id == user_id)).scalar()
        Jobs = db.query(syncingJob).filter(syncingJob.user_id == user_id).all()
        Total_KnowldegeStores_Syncing=0
        for store in knowledge_stores:
            if store.syncing_state:
                Total_KnowldegeStores_Syncing+=1

        response_data = {
            "Total_Knowledge_Bases": total_knowledge_bases,
            "Total_Syncing_Jobs": len(Jobs),
            "Total_Knowledge_Store_in_Sycning_State": Total_KnowldegeStores_Syncing,
        }
        db.close()
        return {"status": "ok", "message": "KnowledgeBase Dashboard returned Successfully.", "data": response_data}
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}
def num_tokens_from_string(string: str, encoding_name="text-embedding-ada-002") -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

@app.post("/GetLeadsGeneratedWithTime")
async def GetLeadsGeneratedWithTime(user_id: str,timeframe: str):
    try:
        db = SessionLocal()
        chatbots = db.query(ChatBotsConfigurations).filter(ChatBotsConfigurations.user_id == user_id).all()
        if not chatbots:
            raise HTTPException(status_code=404, detail="Chatbots not found under user ID")
        chatbotIds = []
        for chatbot in chatbots:
            chatbotIds.append(chatbot.chatbot_id)

        LeadswithtimeForChart = db.query(LeadsGenerated.generated_leads_id, LeadsGenerated.created_at).filter(
            LeadsGenerated.chatbot_id.in_(chatbotIds)).all()

        LeadswithtimeForChart_as_strings = [
            (generated_leads_id, created_at.strftime("%Y-%m-%d %H:%M:%S"))  # Format as desired
            for generated_leads_id, created_at in LeadswithtimeForChart]
       # print("helloo")
        processed_data_for_leads_with_time_period = process_leads(LeadswithtimeForChart_as_strings)
        #print(processed_data_for_leads_with_time_period)
        #print(LeadswithtimeForChart_as_strings)
        result_dict = {}
        for count, date_time in LeadswithtimeForChart_as_strings:
           
            result_dict[date_time] = 1
      
        if timeframe=="all_data":
            response_data = {
                "leads_with_time": result_dict
            }
        else:
            response_data = {
                "leads_with_time": (processed_data_for_leads_with_time_period[timeframe])
            }

        db.close()
        return {"status": "ok", "message": "Leads returned Successfully.", "data": response_data}

    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}

@app.get("/GetChatBotsDashBoardByUserID/{user_id}")
async def GetChatBotsDashBoardByUserID(user_id: str):
        try:
            db = SessionLocal()
            chatbots = db.query(ChatBotsConfigurations).filter(ChatBotsConfigurations.user_id == user_id).all()
            if not chatbots:
                raise HTTPException(status_code=404, detail="Chatbots not found under user ID")
            total_chatbots = len(chatbots)
            total_output_tokens = 0
            total_leadsGenerated = 0
            total_input_tokens = 0
            sumofAllMEssages = 0
            chatbotIds = []
            for chatbot in chatbots:
                chatbotIds.append(chatbot.chatbot_id)
                total_leadsGenerated += db.query(func.count(LeadsGenerated.generated_leads_id).filter(
                    LeadsGenerated.chatbot_id == chatbot.chatbot_id)).scalar()
                chatlogs = db.query(ChatLogs).filter(ChatLogs.chatbot_id == chatbot.chatbot_id).all()
                sumofAllMEssages += len(chatlogs)
                for chatlog in chatlogs:
                    total_output_tokens += num_tokens_from_string(chatlog.AI_Responce)
                    total_input_tokens += num_tokens_from_string(chatlog.Human_Message)
                    total_input_tokens += num_tokens_from_string(chatlog.context)

            totalqureisresponded = db.query(func.count(ChatLogs.Message_id.distinct())).filter(
                ChatLogs.chatbot_id.in_(chatbotIds)).scalar()

            LeadswithtimeForChart = db.query(LeadsGenerated.generated_leads_id, LeadsGenerated.created_at).filter(
                LeadsGenerated.chatbot_id.in_(chatbotIds)).all()
            QuerieswithtimeForChart = db.query(ChatLogs.Message_id, ChatLogs.responded_at).filter(
                ChatLogs.chatbot_id.in_(chatbotIds)).all()

            LeadswithtimeForChart_as_strings = [
                (generated_leads_id, created_at.strftime("%Y-%m-%d %H:%M:%S"))  # Format as desired
                for generated_leads_id, created_at in LeadswithtimeForChart
            ]
            QuerieswithtimeForChart_as_strings = [
                (Message_id, responded_at.strftime("%Y-%m-%d %H:%M:%S"))  # Format as desired
                for Message_id, responded_at in QuerieswithtimeForChart
            ]
            processed_data_for_leads_with_time_period = process_leads(LeadswithtimeForChart_as_strings)
            print(processed_data_for_leads_with_time_period)
            response_data = {
                "Total_Chatbots": total_chatbots,
                "Queries_and_Responces_with_Time":QuerieswithtimeForChart_as_strings,
                "Total_Leads_Generated": total_leadsGenerated,
                "Leads_Generated_Raw_Time":LeadswithtimeForChart_as_strings,
                "Leads_Generated_with_Time_Period": processed_data_for_leads_with_time_period,
                "Total_Queries_Responded": totalqureisresponded,
                "Total_AI_Generated_Tokens": total_output_tokens,
                "Total_Input_Tokens_(Context_and_Question)": total_input_tokens,
                "Average_Number_of_Messages_Per_Chatbot": sumofAllMEssages / total_chatbots,
            }
            db.close()
            return {"status": "ok", "message": "Chatbots Dashboard returned Successfully.", "data": response_data}
        except Exception as e:
            db.close()
            return {"status": "error", "message": str(e), "data": None}
def process_leads(data: List[List]) -> Dict:
    year_data = {}
    month_data = {}
    day_data = {}

    for lead in data:
        lead_id = lead[0]
        timestamp_str = lead[1]
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            year = timestamp.year
            month = timestamp.month
            day = timestamp.day

            # Year data
            year_key = str(year)
            if year_key not in year_data:
                year_data[year_key] = 1
            else:
                year_data[year_key] += 1

            # Month data
            month_key = f"{year}-{month:02}"
            if month_key not in month_data:
                month_data[month_key] = 1
            else:
                month_data[month_key] += 1

            # Day data
            day_key = f"{year}-{month:02}-{day:02}"
            if day_key not in day_data:
                day_data[day_key] = 1
            else:
                day_data[day_key] += 1
        except Exception as e:
            # Handle invalid timestamp format
            print(e)
    processed_data = {
        "year_data": year_data,
        "month_data": month_data,
        "day_data": day_data
    }
    return processed_data
