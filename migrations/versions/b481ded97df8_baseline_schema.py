"""baseline schema

Revision ID: b481ded97df8
Revises: 
Create Date: 2026-09-18 21:19:04.287097

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b481ded97df8'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column(sa.Column('messenger_psid', sa.String(length=100), nullable=True))
        # Named explicitly (uq_users_messenger_psid) instead of leaving it
        # None — SQLite doesn't auto-name constraints, and an unnamed one
        # can't be targeted later by downgrade() or any future migration.
        batch_op.create_unique_constraint('uq_users_messenger_psid', ['messenger_psid'])


def downgrade():
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_constraint('uq_users_messenger_psid', type_='unique')
        batch_op.drop_column('messenger_psid')